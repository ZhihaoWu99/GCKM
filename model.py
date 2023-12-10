import torch
import math
import torch.nn as nn


class GCKLayer(nn.Module):
    def __init__(self, gamma, layer):
        super(GCKLayer, self).__init__()
        self.gamma = gamma
        self.layer = layer

    def rbf_kernel_X(self, X, gamma):
        n = X.shape[0]
        Sij = torch.matmul(X, X.T)
        Si = torch.unsqueeze(torch.diag(Sij), 0).T @ torch.ones(1, n).to(X.device)
        Sj = torch.ones(n, 1).to(X.device) @ torch.unsqueeze(torch.diag(Sij), 0)
        D2 = Si + Sj - 2 * Sij
        K = torch.exp(-D2 * gamma)
        # K[torch.isinf(K)] = 1.
        return K

    def rbf_kernel_K(self, K_t, gamma):
        n = K_t.shape[0]
        s = torch.unsqueeze(torch.diag(K_t), 0)
        D2 = torch.ones(n, 1).to(K_t.device) @ s + s.T @ torch.ones(1, n).to(K_t.device) - 2 * K_t
        K = torch.exp(-D2 * gamma)
        # K[torch.isinf(K)] = 1.
        return K

    def forward(self, adj, inputs):
        if self.layer == 0:
            X_t = adj @ inputs
            K = self.rbf_kernel_X(X_t, self.gamma)
            return K
        else:
            K_t = adj @ inputs @ adj.t()
            K = self.rbf_kernel_K(K_t, self.gamma)
            return K


class GCKM(nn.Module):
    def __init__(self, gamma_list):
        super(GCKM, self).__init__()
        self.model = nn.ModuleList()
        for i, gamma in enumerate(gamma_list):
            self.model.append(GCKLayer(gamma, i))

    def forward(self, adj, X):
        K = X
        for layer in self.model:
            K = layer(adj, K)
        return K


class RFF(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RFF, self).__init__()
        self.out_dim = out_dim
        self.weights = nn.Parameter(torch.randn([in_dim, round(out_dim/2)]), requires_grad=False)

    def forward(self, x):
        x = x.matmul(self.weights)
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        D = self.out_dim
        z = z / math.sqrt(D)
        return z


class GCKLayer_E(nn.Module):
    def __init__(self, gamma, in_dim, out_dim):
        super(GCKLayer_E, self).__init__()
        self.gamma = gamma
        self.map = RFF(in_dim, out_dim)

    def forward(self, adj, inputs):
        Z_t = adj @ inputs
        Z = self.map(Z_t/math.sqrt(self.gamma))
        return Z


class GCKM_E(nn.Module):
    def __init__(self, gamma_list, dim_list, mode):
        super(GCKM_E, self).__init__()
        self.model = nn.ModuleList()
        for i, gamma in enumerate(gamma_list):
            self.model.append(GCKLayer_E(gamma, dim_list[i], dim_list[i+1]))
        self.mode = mode

    def forward(self, adj, X):
        Z = X
        for layer in self.model:
            Z = layer(adj, Z)
        if self.mode == 1:
            return Z @ Z.t()
        elif self.mode == 2:
            return Z


