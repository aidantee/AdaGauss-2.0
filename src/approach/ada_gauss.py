import copy
import random
import torch
import torch.nn.functional as F
import numpy as np

from argparse import ArgumentParser
from itertools import compress
from torch import nn
import torch.utils
from torch.utils.data import Dataset
from torchmetrics import Accuracy

from .mvgb import ClassMemoryDataset, ClassDirectoryDataset
from .models.resnet18 import resnet18
from .models.vit import vit_small
from .models.resnet32 import resnet32
from .incremental_learning import Inc_Learning_Appr
from .criterions.proxy_nca import ProxyNCA
from .criterions.proxy_yolo import ProxyYolo
from .criterions.ce import CE

from torch.distributions.multivariate_normal import MultivariateNormal


class SampledDataset(torch.utils.data.Dataset):
    """ Dataset that samples pseudo prototypes from memorized distributions to train pseudo head """
    def __init__(self, distributions, samples, task_offset):
        self.distributions = distributions
        self.samples = samples
        self.total_classes = task_offset[-1]

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        target = random.randint(0, self.total_classes-1)
        val = self.distributions[target].sample()
        return val, target

## Attention Module Newtwork
class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, old_features):
        scaling_factors = torch.sigmoid(self.fc(old_features))
        return scaling_factors

class AttentionAdapter(nn.Module):
    def __init__(self, feature_dim, multiplier):
        super(AttentionAdapter, self).__init__()
        self.attention = AttentionModule(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, multiplier * feature_dim),
            nn.GELU(),
            nn.Linear(multiplier * feature_dim, feature_dim)
        )

    def forward(self, old_features):
        scaling_factors = self.attention(old_features)
        adjustments = self.mlp(old_features)
        adapted_features = old_features + scaling_factors * adjustments
        return adapted_features

class Appr(Inc_Learning_Appr):
    """Class implementing AdaGauss algorithm"""

    def __init__(self, model, device, nepochs=200, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=1,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, nnet="resnet18", patience=5, fix_bn=False, eval_on_train=False,
                 logger=None, N=10000, alpha=1., lr_backbone=0.01, lr_adapter=0.01, beta=1., distillation="projected", use_224=False, S=64, dump=False, rotation=False, distiller="linear", adapter="linear", criterion="proxy-nca", lamb=10, tau=2, smoothing=0., sval_fraction=0.95,
                 adaptation_strategy="full", pretrained_net=False, normalize=False, shrink=0., multiplier=32, classifier="bayes", 
                 gamma_supcon=1.0, supcon_tau=0.07, samples_per_class=10, gamma_sep=0.0, sep_margin=10.0, sep_pooled_eps=1e-6):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset=None)

        self.N = N
        self.S = S
        self.dump = dump
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.lr_backbone = lr_backbone
        self.lr_adapter = lr_adapter
        self.multiplier = multiplier
        self.shrink = shrink
        self.smoothing = smoothing
        self.adaptation_strategy = adaptation_strategy
        self.old_model = None
        self.pretrained = pretrained_net
        # Inter-class separation loss (AdaGauss+)
        self.gamma_sep = gamma_sep
        self.sep_margin = sep_margin
        self.sep_pooled_eps = sep_pooled_eps
        # running statistics containers for incremental updates (eqs. 5 & 6)
        self.running_means = {}
        self.running_covs = {}
        self.class_counts = {}
        # Supervised contrastive loss using pseudo-prototypes from memorized Gaussians
        self.gamma_supcon = gamma_supcon
        self.supcon_tau = supcon_tau
        self.samples_per_class = samples_per_class

        if nnet == "vit":
            state_dict = torch.load("dino_deitsmall16_pretrain.pth")
            self.model = vit_small(num_features=S)
            self.model.load_state_dict(state_dict, strict=False)
            for name, param in self.model.named_parameters():
                if "blocks.11" not in name:
                    param.requires_grad = False
        elif nnet == "resnet18":
            self.model = resnet18(num_features=S, is_224=use_224)
            if pretrained_net:
                # wget https://download.pytorch.org/models/resnet18-f37072fd.pth
                state_dict = torch.load("../resnet18-f37072fd.pth")
                del state_dict["fc.weight"]
                del state_dict["fc.bias"]
                self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model = resnet32(num_features=S)
            if pretrained_net or use_224:
                raise RuntimeError("No pretrained weights for resnet 32")

        self.model.to(device, non_blocking=True)
        self.train_data_loaders, self.val_data_loaders = [], []
        self.means = torch.empty((0, self.S), device=self.device)
        self.covs = torch.empty((0, self.S, self.S), device=self.device)
        self.covs_raw = torch.empty((0, self.S, self.S), device=self.device)  # not shrinked, not adapted
        self.covs_inverted = None
        self.classifier = classifier
        self.pseudo_head = None
        self.is_normalization = normalize
        self.is_rotation = rotation
        self.task_offset = [0]
        self.classes_in_tasks = []
        self.criterion_type = criterion
        self.criterion = {"proxy-yolo": ProxyYolo,
                          "proxy-nca": ProxyNCA,
                          "ce": CE}[criterion]
        self.heads = torch.nn.ModuleList()
        self.sval_fraction = sval_fraction
        self.svals_explained_by = []
        self.distiller_type = distiller
        self.distillation = distillation
        self.adapter_type = adapter

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--N', help='Number of samples to adapt cov', type=int, default=10000)
        parser.add_argument('--S', help='latent space size', type=int, default=64)
        parser.add_argument('--alpha', help='Weight of anti-collapse loss', type=float, default=1.0)
        parser.add_argument('--beta', help='Anti-collapse loss clamp', type=float, default=1.0)
        parser.add_argument('--lamb', help='Weight of kd loss', type=float, default=10)
        parser.add_argument('--lr-backbone', help='lr for backbone of the pretrained model', type=float, default=0.01)
        parser.add_argument('--lr-adapter', help='lr for backbone of the adapter', type=float, default=0.01)
        parser.add_argument('--multiplier', help='mlp multiplier', type=int, default=32)
        parser.add_argument('--tau', help='temperature for logit distill', type=float, default=2)
        parser.add_argument('--shrink', help='shrink during training', type=float, default=0)
        parser.add_argument('--sval-fraction', help='Fraction of eigenvalues sum that is explained', type=float, default=0.95)
        parser.add_argument('--adaptation-strategy', help='Activation functions in resnet', type=str, choices=["none", "mean", "diag", "full"], default="full")
        parser.add_argument('--distiller', help='Distiller', type=str, choices=["linear", "mlp"], default="mlp")
        parser.add_argument('--adapter', help='Adapter', type=str, choices=["linear", "mlp", "attention"], default="mlp")
        parser.add_argument('--criterion', help='Loss function', type=str, choices=["ce", "proxy-nca", "proxy-yolo"], default="ce")
        parser.add_argument('--nnet', help='Type of neural network', type=str, choices=["vit", "resnet18", "resnet32"], default="resnet18")
        parser.add_argument('--classifier', help='Classifier type', type=str, choices=["linear", "bayes", "nmc"], default="bayes")
        parser.add_argument('--distillation', help='Loss function', type=str, choices=["projected", "logit", "feature", "none"], default="projected")
        parser.add_argument('--smoothing', help='label smoothing', type=float, default=0.0)
        parser.add_argument('--use-224', help='Additional max pool and different conv1 in Resnet18', action='store_true', default=False)
        parser.add_argument('--pretrained-net', help='Load pretrained weights', action='store_true', default=False)
        parser.add_argument('--normalize', help='normalize features and covariance matrices', action='store_true', default=False)
        parser.add_argument('--dump', help='save checkpoints', action='store_true', default=False)
        parser.add_argument('--rotation', help='Rotate images in the first task to enhance feature extractor', action='store_true', default=False)

        ## supcon new SL
        parser.add_argument('--gamma-supcon', help='Weight of supervised contrastive loss (supcon) using pseudo-prototypes', type=float, default=1.0)
        parser.add_argument('--supcon-tau', help='Temperature for supervised contrastive loss', type=float, default=0.07)
        parser.add_argument('--samples-per-class', help='Number of pseudo-prototype samples per class for supcon', type=int, default=10)
        # Inter-class separation loss (AdaGauss+)
        parser.add_argument('--gamma-sep', help='Weight of inter-class separation loss (L_sep)', type=float, default=0.5)
        parser.add_argument('--sep-margin', help='Target margin m for separation loss D^2_ij < m', type=float, default=10.0)
        parser.add_argument('--sep-pooled-eps', help='Small epsilon added to pooled covariance for invertibility', type=float, default=1e-6)

        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        num_classes_in_t = len(np.unique(trn_loader.dataset.labels))
        self.classes_in_tasks.append(num_classes_in_t)
        self.train_data_loaders.extend([trn_loader])
        self.val_data_loaders.extend([val_loader])
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        self.task_offset.append(num_classes_in_t + self.task_offset[-1])
        print("### Training backbone ###")
        # state_dict = torch.load(f"../ckpts/model_{t}.pth")
        # self.model.load_state_dict(state_dict, strict=True)
        self.train_backbone(t, trn_loader, val_loader, num_classes_in_t)
        if self.dump:
            torch.save(self.model.state_dict(), f"{self.logger.exp_path}/model_{t}.pth")
        if t > 0 and self.adaptation_strategy != "none":
            print("### Adapting prototypes ###")
            self.adapt_distributions(t, trn_loader, val_loader)
        print("### Creating new prototypes ###\n")
        self.create_distributions(t, trn_loader, val_loader, num_classes_in_t)

        # Calculate inverted covariances for evaluation with mahalanobis
        covs = self.covs.clone()
        print(f"Cov matrix det: {torch.linalg.det(covs)}")
        for i in range(covs.shape[0]):
            print(f"Rank for class {i}: {torch.linalg.matrix_rank(self.covs_raw[i], tol=0.01)}, {torch.linalg.matrix_rank(self.covs[i], tol=0.01)}")
            covs[i] = self.shrink_cov(covs[i], 3)
        if self.is_normalization:
            covs = self.norm_cov(covs)
        self.covs_inverted = torch.inverse(covs)

        # sampled_protos_norms = []
        # for c in range(self.means.shape[0]):
        #     cov = self.covs[c].clone()
        #     distribution = MultivariateNormal(self.means[c], cov)
        #     samples = distribution.sample((self.N,))
        #     sampled_protos_norms.append(float(samples.norm(dim=1).mean()))
        # sampled_protos_norms = np.array(sampled_protos_norms)
        # sampled_norm_per_task = []
        # for i in range(len(self.task_offset[:-1])):
        #     mean = np.mean(sampled_protos_norms[self.task_offset[i]:self.task_offset[i+1]])
        #     sampled_norm_per_task.append(mean)
        # print(f"Norm of pseudoprototypes {sampled_norm_per_task}")

        if self.classifier == "linear":
            self.train_linear_head(t)

        self.check_singular_values(t, val_loader)
        self.print_singular_values()
        self.print_covs(trn_loader, val_loader)
        self.print_mahalanobis(t)

    def train_backbone(self, t, trn_loader, val_loader, num_classes_in_t):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        print(f'The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters')
        print(f'The expert has {sum(p.numel() for p in self.model.parameters() if not p.requires_grad):,} shared parameters\n')
        distiller = nn.Linear(self.S, self.S)
        if self.distiller_type == "mlp":
            distiller = nn.Sequential(nn.Linear(self.S, self.multiplier * self.S),
                                      nn.GELU(),
                                      nn.Linear(self.multiplier * self.S, self.S)
                                      )

        distiller.to(self.device, non_blocking=True)
        criterion = self.criterion(num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
        if t == 0 and self.is_rotation:
            criterion = self.criterion(4*num_classes_in_t, self.S, self.device, smoothing=self.smoothing)
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size // 4, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size // 4, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        self.heads.eval()
        old_heads = copy.deepcopy(self.heads)
        parameters = list(self.model.parameters()) + list(criterion.parameters()) + list(distiller.parameters()) + list(self.heads.parameters())
        parameters_dict = [
            {"params": list(self.model.parameters())[:-1], "lr": self.lr_backbone},
            {"params": list(criterion.parameters()) + list(self.model.parameters())[-1:]},
            {"params": list(distiller.parameters())},
            {"params": list(self.heads.parameters())},
        ]
        optimizer, lr_scheduler = self.get_optimizer(parameters_dict if self.pretrained else parameters, t, self.wd)

        for epoch in range(self.nepochs):
            train_loss, train_kd_loss, valid_loss, valid_kd_loss = [], [], [], []
            train_ac, train_determinant = [], []
            train_hits, val_hits, train_total, val_total = 0, 0, 0, 0
            self.model.train()
            criterion.train()
            distiller.train()
            for images, targets in trn_loader:
                if t == 0 and self.is_rotation:
                    images, targets = compute_rotations(images, targets, num_classes_in_t)
                targets -= self.task_offset[t]
                bsz = images.shape[0]
                train_total += bsz
                images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                features = self.model(images)
                if epoch < int(self.nepochs * 0.01):
                    features = features.detach()
                loss, logits = criterion(features, targets)

                if self.distillation == "logit":
                    total_loss, kd_loss = self.distill_logits(t, loss, features, images, old_heads)
                elif self.distillation == "projected":
                    total_loss, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                elif self.distillation == "feature":
                    total_loss, kd_loss = self.distill_features(t, loss, features, images)
                else:  # no distillation
                    total_loss, kd_loss = loss, 0.

                ac, det = 0, torch.tensor(0)
                if self.alpha > 0:
                    ac, det = loss_ac(features, self.beta)
                    total_loss += self.alpha * ac

                # Inter-class separation loss (AdaGauss+)
                if t > 0 and self.gamma_sep > 0.0:
                    # absolute labels for current mini-batch
                    abs_targets = targets + self.task_offset[t]
                    # update running stats per class using this minibatch (eqs. 5 & 6)
                    self.update_running_stats_for_batch(features.detach(), abs_targets)
                    # compute separation loss from running stats (this returns tensor with grad=None)
                    sep_val = self.sep_loss_from_running_stats(margin=self.sep_margin, eps=self.sep_pooled_eps)
                    total_loss = total_loss + self.gamma_sep * sep_val

                # Supervised contrastive loss using pseudo-prototypes from memorized Gaussians
                if t > 0 and self.gamma_supcon > 0.0:
                    # number of pseudo samples per old class (paper typically uses 10)
                    samples_per_class = self.samples_per_class
                    old_class_count = self.task_offset[t]
                    if old_class_count > 0:
                        pseudo_samples = []
                        pseudo_labels = []
                        for c in range(old_class_count):
                            cov = self.covs[c].clone()
                            mv = MultivariateNormal(self.means[c], cov)
                            s = mv.sample((samples_per_class,)).to(self.device)
                            pseudo_samples.append(s)
                            pseudo_labels.extend([c] * samples_per_class)
                        pseudo_samples = torch.cat(pseudo_samples, dim=0)
                        pseudo_labels = torch.tensor(pseudo_labels, device=self.device)
                        # absolute labels for current mini-batch
                        abs_targets = targets + self.task_offset[t]
                        z_img = F.normalize(features, p=2, dim=1)
                        z_pseudo = F.normalize(pseudo_samples, p=2, dim=1)
                        z_all = torch.cat([z_img, z_pseudo], dim=0)
                        labels_all = torch.cat([abs_targets, pseudo_labels], dim=0)
                        supcon_val = supcon_loss(z_all, labels_all, self.supcon_tau)
                        total_loss = total_loss + self.gamma_supcon * supcon_val
                            
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1)
                optimizer.step()
                if logits is not None:
                    train_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                train_loss.append(float(bsz * loss))
                train_ac.append(float(ac))
                train_determinant.append(float(torch.clamp(torch.abs(det), max=1e8)))
                train_kd_loss.append(float(bsz * kd_loss))
            lr_scheduler.step()

            val_total = 1e-8
            if epoch % 10 == 9:
                self.model.eval()
                criterion.eval()
                distiller.eval()
                with torch.no_grad():
                    for images, targets in val_loader:
                        if t == 0 and self.is_rotation:
                            images, targets = compute_rotations(images, targets, num_classes_in_t)
                        targets -= self.task_offset[t]
                        bsz = images.shape[0]
                        val_total += bsz
                        images, targets = images.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                        features = self.model(images)
                        loss, logits = criterion(features, targets)
                        if self.distillation == "logit":
                            _, kd_loss = self.distill_logits(t, loss, features, images, old_heads)
                        elif self.distillation == "projected":
                            _, kd_loss = self.distill_projected(t, loss, features, distiller, images)
                        elif self.distillation == "feature":
                            _, kd_loss = self.distill_features(t, loss, features, images)
                        else:  # no distillation
                            kd_loss = 0.

                        if logits is not None:
                            val_hits += float(torch.sum((torch.argmax(logits, dim=1) == targets)))
                        valid_loss.append(float(bsz * loss))
                        valid_kd_loss.append(float(bsz * kd_loss))

            train_loss = sum(train_loss) / train_total
            train_kd_loss = sum(train_kd_loss) / train_total
            train_determinant = sum(train_determinant) / len(train_determinant)
            valid_loss = sum(valid_loss) / val_total
            valid_kd_loss = sum(valid_kd_loss) / val_total
            train_ac = sum(train_ac) / len(train_ac)
            train_acc = train_hits / train_total
            val_acc = val_hits / val_total

            print(f"Epoch: {epoch} Train: {train_loss:.2f} KD: {train_kd_loss:.3f} Acc: {100 * train_acc:.2f} Singularity: {train_ac:.3f} Det: {train_determinant:.5f} "
                  f"Val: {valid_loss:.2f} KD: {valid_kd_loss:.3f} Acc: {100 * val_acc:.2f}")

        if self.distillation == "logit":
            self.heads.append(criterion.head)

    def update_running_stats_for_batch(self, features, abs_labels):
        """Update per-class running means and covariances using equations (5) and (6).
        features: (B, D) tensor (can be detached)
        abs_labels: (B,) absolute class ids
        """
        device = features.device
        unique = torch.unique(abs_labels)
        for c in unique:
            c_int = int(c.item())
            mask = (abs_labels == c)
            x_c = features[mask]
            n_batch = x_c.size(0)
            if n_batch == 0:
                continue
            x_bar = x_c.mean(dim=0)
            if n_batch > 1:
                centered = x_c - x_bar.unsqueeze(0)
                Sigma_batch = (centered.T @ centered) / (n_batch - 1)
            else:
                Sigma_batch = torch.zeros((features.size(1), features.size(1)), device=device)

            if c_int not in self.running_means:
                self.running_means[c_int] = x_bar.clone()
                self.running_covs[c_int] = Sigma_batch.clone()
                self.class_counts[c_int] = n_batch
            else:
                n_c = self.class_counts[c_int]
                mu_old = self.running_means[c_int]
                # eq (5)
                self.running_means[c_int] = mu_old + (n_batch / (n_c + n_batch)) * (x_bar - mu_old)
                # eq (6)
                if (n_c + n_batch - 2) > 0:
                    Sigma_old = self.running_covs[c_int]
                    numerator = (n_c - 1) * Sigma_old + (n_batch - 1) * Sigma_batch
                    self.running_covs[c_int] = numerator / (n_c + n_batch - 2)
                # update counts
                self.class_counts[c_int] = n_c + n_batch

    def sep_loss_from_running_stats(self, margin=10.0, eps=1e-6):
        """Compute L_sep from the current running_means and running_covs dictionaries.
        Returns a scalar tensor (with grad_fn if possible).
        """
        if len(self.running_means) < 2:
            # return zero that won't break backward
            if len(self.running_means) == 1:
                return next(iter(self.running_means.values())).sum() * 0.0
            return torch.tensor(0., device=self.device)

        ids = sorted(self.running_means.keys())
        means = torch.stack([self.running_means[i] for i in ids], dim=0)  # (C, D)
        covs = torch.stack([self.running_covs[i] for i in ids], dim=0)    # (C, D, D)

        pooled = covs.mean(dim=0) + eps * torch.eye(covs.size(1), device=self.device)
        pooled_inv = torch.inverse(pooled)
        A = means @ pooled_inv @ means.T
        diagA = torch.diag(A).unsqueeze(1)
        D2 = diagA + diagA.T - 2.0 * A
        i_idx, j_idx = torch.triu_indices(D2.size(0), D2.size(0), offset=1)
        pair_d2 = D2[i_idx, j_idx]
        hinge = F.relu(margin - pair_d2)
        if hinge.numel() == 0:
            return means.sum() * 0.0
        return hinge.mean()

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader, num_classes_in_t):
        """ Creating distributions for task t"""
        self.model.eval()
        transforms = val_loader.dataset.transform
        new_means = torch.zeros((num_classes_in_t, self.S), device=self.device)
        new_covs = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)
        new_covs_not_shrinked = torch.zeros((num_classes_in_t, self.S, self.S), device=self.device)
        # svals_task = torch.full((10, self.S), fill_value=0., device=self.device)
        for c in range(num_classes_in_t):
            train_indices = torch.tensor(trn_loader.dataset.labels) == c + self.task_offset[t]
            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(trn_loader.dataset.images, train_indices))
                ds = ClassDirectoryDataset(train_images, transforms)
            else:
                ds = trn_loader.dataset.images[train_indices]
                ds = ClassMemoryDataset(ds, transforms)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_+bsz] = features
                features = self.model(torch.flip(images, dims=(3,)))
                class_features[from_+bsz: from_+2*bsz] = features
                from_ += 2*bsz

            # svals = torch.linalg.svdvals(class_features)
            # torch.sort(svals, descending=True)
            # svals_task[c] = svals

            # Calculate  mean and cov
            new_means[c] = class_features.mean(dim=0)
            new_covs[c] = self.shrink_cov(torch.cov(class_features.T), self.shrink)
            new_covs_not_shrinked[c] = torch.cov(class_features.T)
            if self.adaptation_strategy == "diag":
                new_covs[c] = torch.diag(torch.diag(new_covs[c]))

            if torch.isnan(new_covs[c]).any():
                raise RuntimeError(f"Nan in covariance matrix of class {c}")

        # np.savetxt("svals_collapse.txt", np.array(svals_task.mean(0).cpu()))
        self.means = torch.cat((self.means, new_means), dim=0)
        self.covs = torch.cat((self.covs, new_covs), dim=0)
        self.covs_raw = torch.cat((self.covs_raw, new_covs_not_shrinked), dim=0)

    def adapt_distributions(self, t, trn_loader, val_loader):
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, num_workers=trn_loader.num_workers, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size, num_workers=val_loader.num_workers, shuffle=False, drop_last=True)
        # Train the adapter
        self.model.eval()
        adapter = nn.Linear(self.S, self.S)
        if self.adapter_type == "mlp":
            adapter = nn.Sequential(nn.Linear(self.S, self.multiplier * self.S),
                                    nn.GELU(),
                                    nn.Linear(self.multiplier * self.S, self.S)
                                    )
        if self.adapter_type == "attention":
            adapter = AttentionAdapter(self.S, self.multiplier)

        adapter.to(self.device, non_blocking=True)
        optimizer, lr_scheduler = self.get_adapter_optimizer(adapter.parameters())
        old_means = copy.deepcopy(self.means)
        old_covs = copy.deepcopy(self.covs)
        for epoch in range(self.nepochs // 2):
            adapter.train()
            train_loss, valid_loss = [], []
            train_ac, train_determinant = [], []
            for images, _ in trn_loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.no_grad():
                    target = self.model(images)
                    old_features = self.old_model(images)
                adapted_features = adapter(old_features)
                # Compute per-sample squared norms then average over the batch:
                # L = (1/|B|) * sum_i ||F_adapted(x_i) - F_t(x_i)||^2
                diff = adapted_features - target
                loss = torch.mean(torch.sum(diff * diff, dim=1))
                ac, det = 0, torch.tensor(0)
                if self.alpha > 0:
                    ac, det = loss_ac(adapted_features, self.beta)
                total_loss = loss + self.alpha * ac
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1)
                optimizer.step()
                train_loss.append(float(bsz * loss))
                train_ac.append(float(ac))
                train_determinant.append(float(torch.clamp(torch.abs(det), max=1e8)))
            lr_scheduler.step()

            if epoch % 10 == 9:
                adapter.eval()
                with torch.no_grad():
                    for images, _ in val_loader:
                        bsz = images.shape[0]
                        images = images.to(self.device, non_blocking=True)
                        target = self.model(images)
                        old_features = self.old_model(images)
                        adapted_features = adapter(old_features)
                        diff = adapted_features - target
                        total_loss = torch.mean(torch.sum(diff * diff, dim=1))
                        valid_loss.append(float(bsz * total_loss))

            train_loss = sum(train_loss) / len(trn_loader.dataset)
            train_determinant = sum(train_determinant) / len(train_determinant)
            train_ac = sum(train_ac) / len(train_ac)
            valid_loss = sum(valid_loss) / len(val_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f} Val loss: {valid_loss:.2f} Singularity: {train_ac:.3f} Det: {train_determinant:.5f}")

        if self.dump:
            torch.save(adapter.state_dict(), f"{self.logger.exp_path}/adapter_{t}.pth")

        # Adapt
        with torch.no_grad():
            adapter.eval()
            if self.adaptation_strategy == "mean":
                self.means = adapter(self.means)

            if self.adaptation_strategy == "full" or self.adaptation_strategy == "diag":
                for c in range(self.means.shape[0]):
                    cov = self.covs[c].clone()
                    distribution = MultivariateNormal(old_means[c], cov)
                    samples = distribution.sample((self.N,))
                    if torch.isnan(samples).any():
                        raise RuntimeError(f"Nan in features sampled for class {c}")
                    adapted_samples = adapter(samples)
                    self.means[c] = adapted_samples.mean(0)
                    # print(f"Rank pre-adapt {c}: {torch.linalg.matrix_rank(self.covs[c])}")
                    self.covs[c] = torch.cov(adapted_samples.T)
                    self.covs[c] = self.shrink_cov(self.covs[c], self.shrink)
                    if self.adaptation_strategy == "diag":
                        self.covs[c] = torch.diag(torch.diag(self.covs[c]))

            print("### Adaptation evaluation ###")
            for (subset, loaders) in [("train", self.train_data_loaders), ("val", self.val_data_loaders)]:
                old_mean_diff, new_mean_diff = [], []
                old_kld, new_kld = [], []
                old_cov_diff, old_cov_norm_diff, new_cov_diff, new_cov_norm_diff = [], [], [], []
                class_images = np.concatenate([dl.dataset.images for dl in loaders[-2:-1]])
                labels = np.concatenate([dl.dataset.labels for dl in loaders[-2:-1]])

                for c in list(np.unique(labels)):
                    train_indices = torch.tensor(labels) == c

                    if isinstance(trn_loader.dataset.images, list):
                        train_images = list(compress(class_images, train_indices))
                        ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
                    else:
                        ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
                    loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
                    from_ = 0
                    class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
                    for images in loader:
                        bsz = images.shape[0]
                        images = images.to(self.device, non_blocking=True)
                        features = self.model(images)
                        class_features[from_: from_+bsz] = features
                        features = self.model(torch.flip(images, dims=(3,)))
                        class_features[from_+bsz: from_+2*bsz] = features
                        from_ += 2*bsz

                    gt_mean = class_features.mean(0)
                    gt_cov = torch.cov(class_features.T)
                    gt_cov = self.shrink_cov(gt_cov, self.shrink)
                    gt_gauss = torch.distributions.MultivariateNormal(gt_mean, gt_cov)
                    if self.adaptation_strategy == "diag":
                        gt_cov = torch.diag(torch.diag(gt_cov))

                    # Calculate old diffs
                    old_mean_diff.append((gt_mean - old_means[c]).norm())
                    old_cov_diff.append(torch.norm(gt_cov - old_covs[c]))
                    old_cov_norm_diff.append(torch.norm(self.norm_cov(gt_cov.unsqueeze(0)) - self.norm_cov(old_covs[c].unsqueeze(0))))
                    old_gauss = torch.distributions.MultivariateNormal(old_means[c], old_covs[c])
                    old_kld.append(torch.distributions.kl_divergence(old_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, old_gauss))
                    # Calculate new diffs
                    new_mean_diff.append((gt_mean - self.means[c]).norm())
                    new_cov_diff.append(torch.norm(gt_cov - self.covs[c]))
                    new_cov_norm_diff.append(torch.norm(self.norm_cov(gt_cov.unsqueeze(0)) - self.norm_cov(self.covs[c].unsqueeze(0))))
                    new_gauss = torch.distributions.MultivariateNormal(self.means[c], self.covs[c])
                    new_kld.append(torch.distributions.kl_divergence(new_gauss, gt_gauss) + torch.distributions.kl_divergence(gt_gauss, new_gauss))

                old_mean_diff = torch.stack(old_mean_diff)
                old_cov_diff = torch.stack(old_cov_diff)
                old_cov_norm_diff = torch.stack(old_cov_norm_diff)
                old_kld = torch.stack(old_kld)

                new_mean_diff = torch.stack(new_mean_diff)
                new_cov_diff = torch.stack(new_cov_diff)
                new_cov_norm_diff = torch.stack(new_cov_norm_diff)
                new_kld = torch.stack(new_kld)
                print(f"Old {subset} mean diff: {old_mean_diff.mean():.2f} ± {old_mean_diff.std():.2f}")
                print(f"New {subset} mean diff: {new_mean_diff.mean():.2f} ± {new_mean_diff.std():.2f}")
                print(f"Old {subset} cov diff: {old_cov_diff.mean():.2f} ± {old_cov_diff.std():.2f}")
                print(f"New {subset} cov diff: {new_cov_diff.mean():.2f} ± {new_cov_diff.std():.2f}")
                print(f"Old {subset} norm-cov diff: {old_cov_norm_diff.mean():.2f} ± {old_cov_norm_diff.std():.2f}")
                print(f"New {subset} norm-cov diff: {new_cov_norm_diff.mean():.2f} ± {new_cov_norm_diff.std():.2f}")
                print(f"Old {subset} KLD: {old_kld.mean():.2f} ± {old_kld.std():.2f}")
                print(f"New {subset} KLD: {new_kld.mean():.2f} ± {new_kld.std():.2f}")
                print("")

    def distill_projected(self, t, loss, features, distiller, images):
        """ Projected distillation through the distiller, like in https://arxiv.org/abs/2308.12112"""
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
        kd_loss = F.mse_loss(distiller(features), old_features)
        total_loss = loss + self.lamb * kd_loss
        return total_loss, kd_loss

    def distill_features(self, t, loss, features, images):
        """ Feature distillation performed in the latent space of the feature extractor """
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
        kd_loss = F.mse_loss(features, old_features)
        total_loss = loss + self.lamb * kd_loss
        return total_loss, kd_loss

    def distill_logits(self, t, loss, features, images, old_heads):
        """ Logit distillation like in LwF method"""
        if t == 0:
            return loss, 0
        with torch.no_grad():
            old_features = self.old_model(images)
            old_logits = torch.cat([head(old_features) for head in old_heads], dim=1)
        new_logits = torch.cat([head(features) for head in self.heads], dim=1)
        kd_loss = self.cross_entropy(new_logits, old_logits, exp=1 / self.tau)
        total_loss = loss + self.lamb * kd_loss
        return total_loss, kd_loss

    def get_optimizer(self, parameters, t, wd):
        """Returns the optimizer"""
        milestones = (int(0.3*self.nepochs), int(0.6*self.nepochs), int(0.9*self.nepochs))
        lr = self.lr
        if t > 0 and not self.pretrained:
            lr *= 0.1
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=wd, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_adapter_optimizer(self, parameters, milestones=(30, 60, 90)):
        """Returns the optimizer"""
        optimizer = torch.optim.SGD(parameters, lr=self.lr_adapter, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    def get_pseudo_head_optimizer(self, parameters, milestones=(15,)):
        optimizer = torch.optim.SGD(parameters, lr=0.1, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.1)
        return optimizer, scheduler

    @torch.no_grad()
    def eval(self, t, val_loader):
        """ Perform classification using mahalanobis distance OR nearest mean OR linear head. """
        self.model.eval()
        tag_acc = Accuracy("multiclass", num_classes=self.means.shape[0])
        taw_acc = Accuracy("multiclass", num_classes=self.classes_in_tasks[t])
        offset = self.task_offset[t]
        for images, target in val_loader:
            images = images.to(self.device, non_blocking=True)
            features = self.model(images)
            if self.classifier == "linear":
                logits = self.pseudo_head(features)
                tag_preds = torch.argmax(logits, dim=1)
                taw_preds = torch.argmax(logits[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset
            else:
                if self.classifier == "bayes":  # Calculate Mahalanobis distances
                    if self.is_normalization:
                        diff = F.normalize(features.unsqueeze(1), p=2, dim=-1) - F.normalize(self.means.unsqueeze(0), p=2, dim=-1)
                    else:
                        diff = features.unsqueeze(1) - self.means.unsqueeze(0)
                    res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                    res = res @ diff.unsqueeze(3)
                    dist = res.squeeze(2).squeeze(2)
                else:  # Euclidean
                    dist = torch.cdist(features, self.means)
                tag_preds = torch.argmin(dist, dim=1)
                taw_preds = torch.argmin(dist[:, offset: offset + self.classes_in_tasks[t]], dim=1) + offset

            tag_acc.update(tag_preds.cpu(), target)
            taw_acc.update(taw_preds.cpu(), target)

        return 0, float(taw_acc.compute()), float(tag_acc.compute())

    def train_linear_head(self, t):
        """ This is alternative to Bayes and NMC classifier """
        distributions = []
        for c in range(self.means.shape[0]):
            cov = self.covs[c].clone()
            distributions.append(MultivariateNormal(self.means[c], cov))
        dataset = SampledDataset(distributions, 10000, self.task_offset)
        trn_loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0, shuffle=True)
        # Train the adapter
        head = nn.Linear(self.S, self.task_offset[-1])
        head = head.to(self.device)
        optimizer, lr_scheduler = self.get_pseudo_head_optimizer(head.parameters())

        for epoch in range(30):
            head.train()
            train_loss, valid_loss = [], []
            for features, targets in trn_loader:
                bsz = features.shape[0]
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                logits = head(features)
                loss = torch.nn.functional.cross_entropy(logits, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
                optimizer.step()
                train_loss.append(float(bsz * loss))
            lr_scheduler.step()
            train_loss = sum(train_loss) / len(trn_loader.dataset)
            print(f"Epoch: {epoch} Train loss: {train_loss:.2f}")

        with torch.no_grad():
            head.eval()
            self.pseudo_head = head

            logits_per_class = torch.zeros((0, self.means.shape[0]), device=self.device)
            for val_loader in self.val_data_loaders:
                for images, targets in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    features = self.model(images)
                    logits = self.pseudo_head(features)
                    logits_per_class = torch.cat((logits_per_class, logits), dim=0)

            logits_per_task = []
            for i in range(t+1):
                logits_per_task.append(float(logits_per_class[:, self.task_offset[i]:self.task_offset[i+1]].mean()))

            print(f"Logits per task: {list(logits_per_task)}")



    @torch.no_grad()
    def check_singular_values(self, t, val_loader):
        self.model.eval()
        self.svals_explained_by.append([])
        for i, _ in enumerate(self.train_data_loaders):
            if isinstance(self.train_data_loaders[i].dataset.images, list):
                train_images = self.train_data_loaders[i].dataset.images
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(self.train_data_loaders[i].dataset.images, val_loader.dataset.transform)

            loader = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=val_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((len(ds), self.S), fill_value=-999999999.0, device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_ + bsz] = features
                from_ += bsz

            cov = torch.cov(class_features.T)
            svals = torch.linalg.svdvals(cov)
            xd = torch.cumsum(svals, 0)
            xd = xd[xd < self.sval_fraction * torch.sum(svals)]
            explain = xd.shape[0]
            self.svals_explained_by[t].append(explain)

    @torch.no_grad()
    def print_singular_values(self):
        print(f"### {self.sval_fraction} of eigenvalues sum is explained by: ###")
        for t, explained_by in enumerate(self.svals_explained_by):
            print(f"Task {t}: {explained_by}")

    @torch.no_grad()
    def shrink_cov(self, cov, alpha1=1., alpha2=0.):
        if alpha2 == -1.:
            return cov + alpha1 * torch.eye(cov.shape[0], device=self.device)  # ordinary epsilon
        diag_mean = torch.mean(torch.diagonal(cov))
        iden = torch.eye(cov.shape[0], device=self.device)
        mask = iden == 0.0
        off_diag_mean = torch.mean(cov[mask])
        return cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))

    @torch.no_grad()
    def norm_cov(self, cov):
        diag = torch.diagonal(cov, dim1=1, dim2=2)
        std = torch.sqrt(diag)
        cov = cov / (std.unsqueeze(2) @ std.unsqueeze(1))
        return cov

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    @torch.no_grad()
    def print_covs(self, trn_loader, val_loader):
        self.model.eval()
        print("### Norms per task: ###")
        gt_means, gt_covs, gt_inverted_covs = [], [], []
        class_images = np.concatenate([dl.dataset.images for dl in self.train_data_loaders])
        labels = np.concatenate([dl.dataset.labels for dl in self.train_data_loaders])

        # Calculate ground truth
        for c in list(np.unique(labels)):
            train_indices = torch.tensor(labels) == c

            if isinstance(trn_loader.dataset.images, list):
                train_images = list(compress(class_images, train_indices))
                ds = ClassDirectoryDataset(train_images, val_loader.dataset.transform)
            else:
                ds = ClassMemoryDataset(class_images[train_indices], val_loader.dataset.transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=trn_loader.num_workers, shuffle=False)
            from_ = 0
            class_features = torch.full((2 * len(ds), self.S), fill_value=0., device=self.device)
            for images in loader:
                bsz = images.shape[0]
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)
                class_features[from_: from_ + bsz] = features
                features = self.model(torch.flip(images, dims=(3,)))
                class_features[from_ + bsz: from_ + 2 * bsz] = features
                from_ += 2 * bsz

            gt_means.append(class_features.mean(0))
            cov = torch.cov(class_features.T)
            gt_covs.append(cov)
            gt_inverted_covs.append(torch.inverse(self.shrink_cov(cov, self.shrink)))

        gt_means = torch.stack(gt_means)
        gt_covs = torch.stack(gt_covs)
        gt_inverted_covs = torch.stack(gt_inverted_covs)

        # Calculate statistics per task
        mean_norms, cov_norms = [], []
        gt_mean_norms, gt_cov_norms = [], []
        inverted_cov_norms, gt_inverted_cov_norms = [], []
        for task in range(len(self.task_offset[1:])):
            from_ = self.task_offset[task]
            to_ = self.task_offset[task + 1]
            mean_norms.append(round(float(torch.norm(self.means[from_:to_], dim=1).mean()), 2))
            cov_norms.append(round(float(torch.linalg.matrix_norm(self.covs[from_:to_]).mean()), 2))
            inverted_cov_norms.append(round(float(torch.linalg.matrix_norm(torch.inverse(self.covs[from_:to_])).mean()), 2))  # no shrink, no norm!
            gt_mean_norms.append(round(float(torch.norm(gt_means[from_:to_], dim=1).mean()), 2))
            gt_cov_norms.append(round(float(torch.linalg.matrix_norm(gt_covs[from_:to_]).mean()), 2))
            gt_inverted_cov_norms.append(round(float(torch.linalg.matrix_norm(gt_inverted_covs[from_:to_]).mean()), 2))
        print(f"Means: {mean_norms}")
        print(f"GT Means: {gt_mean_norms}")
        print(f"Covs: {cov_norms}")
        print(f"GT Covs: {gt_cov_norms}")
        print(f"Inverted Covs: {inverted_cov_norms}")
        print(f"GT Inverted Covs: {gt_inverted_cov_norms}")

    @torch.no_grad()
    def print_mahalanobis(self, t):
        self.model.eval()
        mahalanobis_per_class = torch.zeros((0, self.means.shape[0]), device=self.device)
        for val_loader in self.val_data_loaders:
            for images, targets in val_loader:
                images = images.to(self.device, non_blocking=True)
                features = self.model(images)

                diff = features.unsqueeze(1) - self.means.unsqueeze(0)
                res = diff.unsqueeze(2) @ self.covs_inverted.unsqueeze(0)
                res = res @ diff.unsqueeze(3)
                dist = res.squeeze(2).squeeze(2)
                mahalanobis_per_class = torch.cat((mahalanobis_per_class, dist), dim=0)

        mahalanobis_per_task = []
        for i in range(t+1):
            mahalanobis_per_task.append(float(mahalanobis_per_class[:, self.task_offset[i]:self.task_offset[i+1]].mean()))

        print(f"Mahalanobis per task: {list(mahalanobis_per_task)}")


def compute_rotations(images, targets, total_classes):
    # compute self-rotation for the first task following PASS https://github.com/Impression2805/CVPR21_PASS
    images_rot = torch.cat([torch.rot90(images, k, (2, 3)) for k in range(1, 4)])
    images = torch.cat((images, images_rot))
    target_rot = torch.cat([(targets + total_classes * k) for k in range(1, 4)])
    targets = torch.cat((targets, target_rot))
    return images, targets


def loss_ac(features, beta):
    cov = torch.cov(features.T)
    cholesky = torch.linalg.cholesky(cov)
    cholesky_diag = torch.diag(cholesky)
    loss = - torch.clamp(cholesky_diag, max=beta).mean()
    # if bool(torch.isinf(loss)) or bool(torch.isnan(loss)):
    #     return torch.tensor(7777.), torch.tensor(0.)
    return loss, torch.det(cov)

def supcon_loss(features, labels, tau=0.1):
    """Supervised contrastive loss (per-sample average) for L2-normalized features.
    features: (N, D) assumed normalized
    labels: (N,) integer labels (absolute)
    tau: temperature
    Returns scalar loss
    """
    device = features.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # NxN
    # remove self-contrast
    diag = torch.eye(mask.size(0), device=device)
    mask = mask * (1 - diag)

    # cosine similarity (since features normalized)
    logits = torch.matmul(features, features.T) / tau
    # numeric stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    exp_logits = torch.exp(logits) * (1 - diag)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    # sum log_prob over positives
    numerator = (mask * log_prob).sum(dim=1)
    positives_per_sample = mask.sum(dim=1)
    # avoid division by zero; compute only for samples with positives
    valid = positives_per_sample > 0
    if valid.sum() == 0:
        return torch.tensor(0., device=device)

    loss_per_sample = - numerator[valid] / (positives_per_sample[valid] + 1e-12)
    return loss_per_sample.mean()
