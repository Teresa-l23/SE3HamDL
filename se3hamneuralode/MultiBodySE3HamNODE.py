# Multi-body extension of SE3HamNODE
# Per-body SE(3) state: [x(3), R(9), v(3), w(3)] = 18 dims
# Full state: [body_1(18), ..., body_N(18), u(u_dim)]

import torch
import numpy as np

from se3hamneuralode import MLP, PSD, MatrixNet
from se3hamneuralode import compute_rotation_matrix_from_quaternion
from .utils import L2_loss

SE3_BODY_DIM = 18  # x(3) + R(9) + v(3) + w(3)


class MultiBodySE3HamNODE(torch.nn.Module):
    """Hamiltonian Neural ODE on SE(3)^N for multi-body systems.

    State layout per body: [x(3), R(9), v_lin(3), w_ang(3)]
    Full input:  [body_1, body_2, ..., body_N, u_control]

    Architecture:
        - Shared M_net1 (linear inertia) applied per body  (input: x_i → 3×3 PSD)
        - Shared M_net2 (rotational inertia) applied per body (input: R_i → 3×3 PSD)
        - Single V_net taking ALL bodies' poses [x_1,R_1,...,x_N,R_N] → scalar
          to capture inter-body coupling (gravity, constraints)
        - Single g_net taking all poses → (N*6) × u_dim actuation matrix
    """

    def __init__(self, n_bodies, device=None, pretrain=True,
                 M_net1=None, M_net2=None, V_net=None, g_net=None,
                 udim=1, init_gain=0.001):
        super(MultiBodySE3HamNODE, self).__init__()
        self.n_bodies = n_bodies
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.body_posedim = self.xdim + self.Rdim   # 12
        self.body_twistdim = self.linveldim + self.angveldim  # 6
        self.body_dim = SE3_BODY_DIM  # 18
        self.udim = udim

        # Total dimensions
        self.total_posedim = n_bodies * self.body_posedim  # N * 12
        self.total_twistdim = n_bodies * self.body_twistdim  # N * 6

        # Shared mass matrix networks (applied per body)
        if M_net1 is None:
            self.M_net1 = PSD(self.xdim, 400, self.linveldim,
                              init_gain=init_gain).to(device)
        else:
            self.M_net1 = M_net1
        if M_net2 is None:
            self.M_net2 = PSD(self.Rdim, 400, self.angveldim,
                              init_gain=init_gain).to(device)
        else:
            self.M_net2 = M_net2

        # V_net takes all poses to capture coupling
        if V_net is None:
            self.V_net = MLP(self.total_posedim, 400, 1,
                             init_gain=init_gain).to(device)
        else:
            self.V_net = V_net

        # g_net maps poses → (total_twist × udim) control matrix
        if g_net is None:
            self.g_net = MatrixNet(
                self.total_posedim, 400,
                self.total_twistdim * self.udim,
                shape=(self.total_twistdim, self.udim),
                init_gain=init_gain).to(device)
        else:
            self.g_net = g_net

        self.device = device
        self.nfe = 0
        if pretrain:
            self.pretrain()

    # --- Pretraining (M_nets to identity) ------------------------------------
    def pretrain(self):
        # Pretrain M_net1 (linear inertia)
        x = np.arange(-10, 10, 0.5)
        n_grid = len(x)
        batch = n_grid ** 3
        xx, yy, zz = np.meshgrid(x, x, x)
        Xgrid = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
        Xgrid = torch.tensor(Xgrid, dtype=torch.float64).to(self.device)

        m_guess = torch.eye(3, dtype=torch.float64, device=self.device)
        m_guess = m_guess.unsqueeze(0).expand(batch, -1, -1)
        optim1 = torch.optim.Adam(self.M_net1.parameters(), 1e-3, foreach=False)
        m_hat = self.M_net1(Xgrid)
        loss = L2_loss(m_hat, m_guess)
        print("MultiBody: Start pretraining Mnet1!", loss.item())
        step = 0
        while loss > 1e-6:
            loss.backward()
            optim1.step()
            optim1.zero_grad()
            m_hat = self.M_net1(Xgrid)
            loss = L2_loss(m_hat, m_guess)
            step += 1
            if step % 10 == 0:
                print("  step", step, loss.item())
        print("MultiBody: Pretraining Mnet1 done!", loss.item())
        del Xgrid; torch.cuda.empty_cache()

        # Pretrain M_net2 (rotational inertia)
        batch = 250000
        rand_ = np.random.uniform(size=(batch, 3))
        u1, u2, u3 = rand_[:, 0], rand_[:, 1], rand_[:, 2]
        quat = np.array([
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3)
        ]).T
        q_tensor = torch.tensor(quat, dtype=torch.float64).to(self.device)
        R_tensor = compute_rotation_matrix_from_quaternion(q_tensor).view(-1, 9)

        inertia_guess = torch.eye(3, dtype=torch.float64, device=self.device)
        inertia_guess = inertia_guess.unsqueeze(0).expand(batch, -1, -1)
        optim2 = torch.optim.Adam(self.M_net2.parameters(), 1e-3, foreach=False)
        m_hat = self.M_net2(R_tensor)
        loss = L2_loss(m_hat, inertia_guess)
        print("MultiBody: Start pretraining Mnet2!", loss.item())
        step = 0
        while loss > 1e-6:
            loss.backward()
            optim2.step()
            optim2.zero_grad()
            m_hat = self.M_net2(R_tensor)
            loss = L2_loss(m_hat, inertia_guess)
            step += 1
            if step % 10 == 0:
                print("  step", step, loss.item())
        print("MultiBody: Pretraining Mnet2 done!", loss.item())
        del q_tensor, R_tensor; torch.cuda.empty_cache()

    # --- Forward pass --------------------------------------------------------
    def forward(self, t, input):
        with torch.enable_grad():
            self.nfe += 1
            N = self.n_bodies
            batch_size = input.shape[0]

            # 1. Split input into per-body states + control
            body_states, u = torch.split(
                input, [N * self.body_dim, self.udim], dim=1)

            # Reshape to (batch, N, 18)
            body_states = body_states.view(batch_size, N, self.body_dim)

            # Split each body: x(3), R(9), v(3), w(3)
            xs = body_states[:, :, :3]                    # (B, N, 3)
            Rs = body_states[:, :, 3:12]                  # (B, N, 9)
            vs = body_states[:, :, 12:15]                 # (B, N, 3)
            ws = body_states[:, :, 15:18]                 # (B, N, 3)

            # 2. Compute momenta per body via shared M_nets
            # Flatten across bodies for batched network forward pass
            xs_flat = xs.reshape(batch_size * N, 3)        # (B*N, 3)
            Rs_flat = Rs.reshape(batch_size * N, 9)        # (B*N, 9)
            vs_flat = vs.reshape(batch_size * N, 3)        # (B*N, 3)
            ws_flat = ws.reshape(batch_size * N, 3)        # (B*N, 3)

            M_inv1 = self.M_net1(xs_flat)                  # (B*N, 3, 3)
            M_inv2 = self.M_net2(Rs_flat)                  # (B*N, 3, 3)

            pv = torch.squeeze(torch.matmul(
                torch.inverse(M_inv1),
                vs_flat.unsqueeze(2)), dim=2)              # (B*N, 3)
            pw = torch.squeeze(torch.matmul(
                torch.inverse(M_inv2),
                ws_flat.unsqueeze(2)), dim=2)              # (B*N, 3)

            # 3. Build concatenated pose and q_p for autograd
            # All poses: [x1, R1, x2, R2, ...]  = (B, N*12)
            all_poses = torch.cat([xs, Rs], dim=2).reshape(
                batch_size, self.total_posedim)

            # q_p for Hamiltonian: [all_poses, pv_1, pw_1, ..., pv_N, pw_N]
            pv_unflt = pv.view(batch_size, N, 3)
            pw_unflt = pw.view(batch_size, N, 3)
            all_momenta = torch.cat([pv_unflt, pw_unflt], dim=2).reshape(
                batch_size, self.total_twistdim)

            q_p = torch.cat([all_poses, all_momenta], dim=1)
            q_p.requires_grad_(True)

            # Re-extract with gradient tracking
            all_poses_g = q_p[:, :self.total_posedim]
            all_mom_g = q_p[:, self.total_posedim:]

            # 4. Network forward passes
            M_inv1 = self.M_net1(xs_flat)   # re-evaluate for autograd
            M_inv2 = self.M_net2(Rs_flat)
            V_q = self.V_net(all_poses_g)
            g_q = self.g_net(all_poses_g)   # (B, total_twist, udim)

            # 5. Compute Hamiltonian
            # H = sum_i [ 0.5 * pv_i^T M_inv1_i pv_i + 0.5 * pw_i^T M_inv2_i pw_i ] + V
            H = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
            for i in range(N):
                pv_i = pv[i * batch_size:(i + 1) * batch_size] if False else \
                       all_mom_g[:, i * 6:i * 6 + 3]
                pw_i = all_mom_g[:, i * 6 + 3:i * 6 + 6]

                Mi1 = M_inv1[i::N]  # every N-th starting from i  -> wrong indexing
                Mi2 = M_inv2[i::N]

                H = H + 0.5 * torch.sum(
                    pv_i * torch.squeeze(torch.matmul(
                        self.M_net1(all_poses_g[:, i * 12:i * 12 + 3]),
                        pv_i.unsqueeze(2)), 2), dim=1)
                H = H + 0.5 * torch.sum(
                    pw_i * torch.squeeze(torch.matmul(
                        self.M_net2(all_poses_g[:, i * 12 + 3:i * 12 + 12]),
                        pw_i.unsqueeze(2)), 2), dim=1)
            H = H + V_q.squeeze()

            # 6. Compute dH/d(q_p) via autograd
            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]

            # 7. Hamilton's equations per body
            dxs_list = []
            dRs_list = []
            dvs_list = []
            dws_list = []

            for i in range(N):
                pose_offset = i * 12
                mom_offset = self.total_posedim + i * 6

                dHdx_i = dH[:, pose_offset:pose_offset + 3]
                dHdR_i = dH[:, pose_offset + 3:pose_offset + 12]
                dHdpv_i = dH[:, mom_offset:mom_offset + 3]
                dHdpw_i = dH[:, mom_offset + 3:mom_offset + 6]

                pv_i = all_mom_g[:, i * 6:i * 6 + 3]
                pw_i = all_mom_g[:, i * 6 + 3:i * 6 + 6]

                Rmat_i = all_poses_g[:, pose_offset + 3:pose_offset + 12].view(-1, 3, 3)
                x_i = all_poses_g[:, pose_offset:pose_offset + 3]

                # SE(3) dynamics (same structure as single-body SE3HamNODE)
                dx_i = torch.squeeze(torch.matmul(Rmat_i, dHdpv_i.unsqueeze(2)))
                dR03 = torch.cross(Rmat_i[:, 0, :], dHdpw_i)
                dR36 = torch.cross(Rmat_i[:, 1, :], dHdpw_i)
                dR69 = torch.cross(Rmat_i[:, 2, :], dHdpw_i)
                dR_i = torch.cat([dR03, dR36, dR69], dim=1)

                # Control force for this body
                F_all = torch.squeeze(torch.matmul(g_q, u.unsqueeze(2)))
                F_i = F_all[:, i * 6:(i + 1) * 6]
                Fv_i = F_i[:, :3]
                Fw_i = F_i[:, 3:]

                dpv_i = (torch.cross(pv_i, dHdpw_i)
                         - torch.squeeze(torch.matmul(Rmat_i.transpose(1, 2),
                                                      dHdx_i.unsqueeze(2)))
                         + Fv_i)
                dpw_i = (torch.cross(pw_i, dHdpw_i)
                         + torch.cross(pv_i, dHdpv_i)
                         + torch.cross(Rmat_i[:, 0, :], dHdR_i[:, 0:3])
                         + torch.cross(Rmat_i[:, 1, :], dHdR_i[:, 3:6])
                         + torch.cross(Rmat_i[:, 2, :], dHdR_i[:, 6:9])
                         + Fw_i)

                # Convert momentum derivatives to velocity derivatives
                Mi1 = self.M_net1(x_i)
                Mi2 = self.M_net2(all_poses_g[:, pose_offset + 3:pose_offset + 12])

                # dM_inv/dt for linear part
                dM_inv_dt1 = torch.zeros_like(Mi1)
                for r in range(3):
                    for c in range(3):
                        dM = torch.autograd.grad(
                            Mi1[:, r, c].sum(), q_p, create_graph=True)[0]
                        dM_inv_dt1[:, r, c] = (dM[:, pose_offset:pose_offset + 3] * dx_i).sum(-1)

                dv_i = (torch.squeeze(torch.matmul(Mi1, dpv_i.unsqueeze(2)), 2)
                        + torch.squeeze(torch.matmul(dM_inv_dt1, pv_i.unsqueeze(2)), 2))

                # dM_inv/dt for angular part
                dM_inv_dt2 = torch.zeros_like(Mi2)
                for r in range(3):
                    for c in range(3):
                        dM = torch.autograd.grad(
                            Mi2[:, r, c].sum(), q_p, create_graph=True)[0]
                        dM_inv_dt2[:, r, c] = (
                            dM[:, pose_offset + 3:pose_offset + 12] * dR_i).sum(-1)

                dw_i = (torch.squeeze(torch.matmul(Mi2, dpw_i.unsqueeze(2)), 2)
                        + torch.squeeze(torch.matmul(dM_inv_dt2, pw_i.unsqueeze(2)), 2))

                dxs_list.append(dx_i)
                dRs_list.append(dR_i)
                dvs_list.append(dv_i)
                dws_list.append(dw_i)

            # 8. Assemble output  [dx1,dR1,dv1,dw1, dx2,dR2,dv2,dw2, ..., 0_u]
            body_derivs = []
            for i in range(N):
                body_derivs.append(torch.cat([
                    dxs_list[i], dRs_list[i], dvs_list[i], dws_list[i]
                ], dim=1))
            zero_u = torch.zeros(batch_size, self.udim,
                                 dtype=torch.float64, device=self.device)
            return torch.cat(body_derivs + [zero_u], dim=1)
