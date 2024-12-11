# -*- coding: utf-8 -*-
# @Time : 2024/4/18 10:39
# @Author : Anonymous
# @E-mail : anonymous@gmail.com
# @Site : 
# @project: vehicle_dispatch
# @File : training_traditional_RL.py
# @Software: PyCharm
import logging
import time

import torch
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pickle as pkl
from config import parser
from model.DSEmb.graphbase import NCModel
from model.RL.layers import Reward, Actor, Critic
from utils.clear_model_logs import clear
from utils.data_utils import DS_data, TrajData
from utils.eval_utils import acc_f1
from utils.train_utils import format_metrics, select_action

if __name__ == "__main__":
    clear('../')

    logging.getLogger().setLevel(logging.ERROR)

    args = parser.parse_args()

    args.log_dir = '../' + args.log_dir
    args.trajectory_pos = '../' + args.trajectory_pos
    args.trajectory_neg = '../' + args.trajectory_neg
    args.trajectory_all = '../' + args.trajectory_all
    args.multigraph_adj = '../' + args.multigraph_adj
    args.multigraph_fea = '../' + args.multigraph_fea

    args.device = torch.device("cpu")
    if args.cuda >= 0:
        args.device = torch.device("cuda:" + str(args.cuda))
    # some other parameters
    clip_param = 0.2
    max_grad_norm_param = 0.5
    gamma_param = 0.99

    ID = 0
    # tensorboardX logbook
    Writer = SummaryWriter(args.log_dir + '/tensorboard/' +  str(ID) + '/')

    # read data
    ## graph data ##
    DSDATA = DS_data(args)
    DS_dataloader = DataLoader(DSDATA, batch_size=4, shuffle=True, num_workers=1)
    ### get node number, node feature dimension
    for i, batch in enumerate(DS_dataloader):
        a, b, c, a_label, b_label, c_label = batch
        args.n_nodes = a.shape[1]
        args.feat_dim = a.shape[2]
        break
    adj_G2, adj_G5, adj_G10 = DSDATA.get_adj()
    fea_G2, fea_G5, fea_G10 = DSDATA.get_all_fea()
    ## trajectory data
    contrast_mode_str = 'all'
    trajdata = TrajData(args, 0.3, contrast_mode='all')
    Traj_dataloader = DataLoader(trajdata, batch_size=args.batch_size, shuffle=True, num_workers=1)
    ### get feature dimension, contrast output dimension
    args.in_dim = -1
    if contrast_mode_str == 'all':
        args.out_dim = 50
        args.avg = 'micro'
    else:
        args.out_dim = 2
        args.avg = 'binary'
    for i, batch in enumerate(Traj_dataloader):
        vid, state, action, reward = batch
        args.in_dim = state.shape[2] + action.shape[2]
        break
    ## positive Trajectory data
    f = open(args.trajectory_pos, 'rb')
    pos_list = pkl.load(f)
    f.close()
    MAX_FEE = 1.0
    for i in range(len(pos_list)):
        if i == 0:
            data = pos_list[i]['data']
            label = pos_list[i]['label']
            static_reward = pos_list[i]['reward']
            MAX_FEE = max(static_reward)
        else:
            data = np.r_[data, pos_list[i]['data']]
            label = np.r_[label, pos_list[i]['label']]
            static_reward = np.r_[static_reward, pos_list[i]['reward']]
            if MAX_FEE < max(static_reward):
                MAX_FEE = max(static_reward)
    # print('MAX_FEE:', MAX_FEE)
    # print('data shape:', data.shape)
    # print('label shape:', label.shape)
    # print('static_reward shape:', static_reward.shape)
    data = torch.tensor(data)
    label = torch.tensor(label)
    static_reward = torch.tensor(static_reward)
    args.num_state = int(data.shape[-1] + args.graph_dim * (adj_G2.shape[0] + adj_G5.shape[0] + adj_G10.shape[0]))
    args.num_ation = int(label.shape[-1] - 1)
    print('num state:', args.num_state)
    print('num ation:', args.num_ation)


    # define graph embedding model
    model_G2 = NCModel(args)
    model_G5 = NCModel(args)
    model_G10 = NCModel(args)
    model_r = Reward(args.in_dim, args.dim, args.out_dim, batch_size=args.batch_size)
    actor = Actor(args.num_state, args.num_ation)
    critic = Critic(args.num_state)


    # model & data to device
    if not args.cuda == -1:
        model_G2 = model_G2.to(args.device)
        model_G5 = model_G5.to(args.device)
        model_G10 = model_G10.to(args.device)
        model_r = model_r.to(args.device)
        actor = actor.to(args.device)
        critic = critic.to(args.device)

        adj_G2 = adj_G2.to(args.device)
        adj_G5 = adj_G5.to(args.device)
        adj_G10 = adj_G10.to(args.device)
        fea_G2 = fea_G2.to(args.device)
        fea_G5 = fea_G5.to(args.device)
        fea_G10 = fea_G10.to(args.device)

        data = data.to(args.device)
        label = label.to(args.device)
        static_reward = static_reward.to(args.device)

    # define learning rate reduce frequency
    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # define optimizer
    opt_G2 = torch.optim.Adam(params=model_G2.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_G5 = torch.optim.Adam(params=model_G5.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_G10 = torch.optim.Adam(params=model_G10.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_r = torch.optim.Adam(params=model_r.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_actor = torch.optim.Adam(params=actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_critic = torch.optim.Adam(params=critic.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler_G2 = torch.optim.lr_scheduler.StepLR(opt_G2,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_G5 = torch.optim.lr_scheduler.StepLR(opt_G5,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_G10 = torch.optim.lr_scheduler.StepLR(opt_G10,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_r = torch.optim.lr_scheduler.StepLR(opt_r,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_actor = torch.optim.lr_scheduler.StepLR(opt_actor,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))
    lr_scheduler_critic = torch.optim.lr_scheduler.StepLR(opt_critic,step_size=int(args.lr_reduce_freq),gamma=float(args.gamma))

    # define loss function
    CEloss = nn.CrossEntropyLoss()
    MSEloss = nn.MSELoss()

    losses_list = []
    global_training_step = 0
    for ep in range(args.epochs):
        print('Epoch:', str(ep+1).zfill(4))

        # graph embedding part
        print('Graph embedding...')
        for i, batch in enumerate(DS_dataloader):
            t = time.time()
            a, b, c, a_label, b_label, c_label = batch
            if not args.cuda == -1:
                a = a.to(args.device)
                b = b.to(args.device)
                c = c.to(args.device)
                a_label = a_label.to(args.device)
                b_label = b_label.to(args.device)
                c_label = c_label.to(args.device)

            model_G2.train()
            opt_G2.zero_grad()
            embeddings_G2 = model_G2.encode(a, adj_G2)
            # print('test train emb shape:', embeddings.shape)
            train_metrics = model_G2.compute_metrics(embeddings_G2, adj_G2, a_label)
            train_metrics['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_G2.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            opt_G2.step()
            lr_scheduler_G2.step()

            model_G5.train()
            opt_G5.zero_grad()
            embeddings_G5 = model_G5.encode(b, adj_G5)
            # print('test train emb shape:', embeddings.shape)
            train_metrics = model_G5.compute_metrics(embeddings_G5, adj_G5, b_label)
            train_metrics['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_G5.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            opt_G5.step()
            lr_scheduler_G5.step()

            model_G10.train()
            opt_G10.zero_grad()
            embeddings_G10 = model_G10.encode(c, adj_G10)
            # print('test train emb shape:', embeddings.shape)
            train_metrics = model_G10.compute_metrics(embeddings_G10, adj_G10, c_label)
            train_metrics['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_G10.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            opt_G10.step()
            lr_scheduler_G10.step()

            if (i + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(i + 1),
                                       'lr_G2: {}'.format(lr_scheduler_G2.get_last_lr()[0]),
                                       'lr_G5: {}'.format(lr_scheduler_G5.get_last_lr()[0]),
                                       'lr_G10: {}'.format(lr_scheduler_G10.get_last_lr()[0]),
                                       format_metrics(train_metrics, 'train'),
                                       'time: {:.4f}s'.format(time.time() - t)
                                       ]))
        model_G10.eval()
        model_G5.eval()
        model_G2.eval()
        print('Graph embedding finished!')
        # got embeddings_G2, embeddings_G5 and embeddings_G10
        print('embedding G2 shape:', embeddings_G2.shape)
        print('embedding G5 shape:', embeddings_G5.shape)
        print('embedding G10 shape:', embeddings_G10.shape)

        # dynamic reward part
        print('Dynamic reward function updating...')
        loss = 0
        acc = 0
        f1 = 0
        cnt = 0
        for i, batch in enumerate(Traj_dataloader):
            t = time.time()
            vid, state, action, reward = batch
            # print('vid:', vid)
            if state.shape[0] < args.batch_size:
                continue
            hidden = model_r.init_hidden()
            if not args.cuda == -1:
                vid = vid.to(args.device)
                state = state.to(args.device)
                action = action.to(args.device)
                hidden = hidden.to(args.device)
            x = torch.cat([state, action], dim=2).float()
            vid = vid.long()

            model_r.train()
            opt_r.zero_grad()
            pred, _ = model_r.forward(x, hidden)
            tmp_loss = CEloss(pred, vid)
            tmp_acc, tmp_f1 = acc_f1(pred, vid, average=args.avg)
            loss += tmp_loss
            acc += tmp_acc
            f1 += tmp_f1
            cnt += 1
            loss.backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model_r.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            opt_r.step()
            lr_scheduler_r.step()

            if (i + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(i + 1),
                                       'lr: {}'.format(lr_scheduler_r.get_last_lr()[0]),
                                       'loss: {}'.format(loss / cnt),
                                       'acc: {}'.format(acc / cnt),
                                       'f1: {}'.format(f1 / cnt),
                                       'time: {:.4f}s'.format(time.time() - t)
                                       ]))
                acc = 0
                f1 = 0
                loss = 0
                cnt = 0
        print('Dynamic reward function updated!')
        # get dynamcic reward
        model_r.eval()
        print('Actor-Critic Network updating...')
        R = 0
        Gt = []
        for ind in range(len(static_reward)-1, -1, -1):
            state = data[ind].unsqueeze(0).unsqueeze(0)
            # print('test state shape:', state.shape)
            true_action = label[ind].unsqueeze(0).unsqueeze(0)
            # print('test action shape:', true_action.shape)
            x = torch.cat([state, true_action], dim=2).float()
            hidden = model_r.init_hidden(1).to(args.device)
            pred, _ = model_r.forward(x, hidden)
            pred = pred.squeeze()
            # print('test pred shape:', pred.shape)
            alpha = 0.05
            R = alpha * (static_reward[ind] / MAX_FEE) + (1 - alpha) * pred[0] + gamma_param * R
            # print('test R:', R)
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(args.device)

        losses = []
        for t in range(data.shape[0]):
            tt = time.time()
            if t == 0:
                actor.train()
                critic.train()
                continue
            graph_fea_G2 = fea_G2[t].unsqueeze(0)
            graph_fea_G5 = fea_G5[t].unsqueeze(0)
            graph_fea_G10 = fea_G10[t].unsqueeze(0)

            G2_emb = model_G2.encode(graph_fea_G2, adj_G2).detach()
            G5_emb = model_G5.encode(graph_fea_G5, adj_G5).detach()
            G10_emb = model_G10.encode(graph_fea_G10, adj_G10).detach()
            G2_emb = G2_emb.view(1, -1)
            G5_emb = G5_emb.view(1, -1)
            G10_emb = G10_emb.view(1, -1)

            state = data[t].unsqueeze(0).to(torch.float32)
            true_action_vec = label[t][:2].unsqueeze(0)
            true_action = label[t][2].unsqueeze(0)

            state = torch.concatenate([state, G2_emb, G5_emb, G10_emb], dim=1)

            # print(state.shape)
            # print(true_action_vec.shape)
            # print(true_action.shape)

            _, old_action_prob = select_action(true_action_vec, true_action)

            Gt_index = Gt[t].view(-1, 1)
            V = critic(state)
            delta = Gt_index - V
            advantage = delta.detach()
            # core part !!!
            out_action_vec = actor(state)  # new policy
            action, new_action_prob = select_action(out_action_vec, true_action)
            a_hat = out_action_vec.detach()
            a  = true_action_vec
            selfdefine_loss = torch.mean(
        torch.minimum(torch.minimum((a_hat[:,1] - a[:,1]) ** 2, ((a_hat[:,1] + 1) - a[:,1]) ** 2), (a_hat[:,1] - (a[:,1] + 1)) ** 2) + (a_hat[:,0] - a[:,0]) ** 2)# torch.mean(torch.minimum(torch.minimum((a_hat - a) ** 2, ((a_hat + 1) - a) ** 2), (a_hat - (a + 1)) ** 2))
            losses.append(selfdefine_loss)
            losses_list.append(selfdefine_loss)

            ratio = new_action_prob / old_action_prob
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantage

            # update actor network
            action_loss = - torch.min(surr1, surr2).mean()
            Writer.add_scalar('action_loss_' + str(ID), action_loss, global_step=global_training_step)
            opt_actor.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm_param)
            opt_actor.step()
            lr_scheduler_actor.step()

            # update critic network
            value_loss = F.mse_loss(Gt_index, V)
            Writer.add_scalar('value_loss_' + str(ID), value_loss, global_step=global_training_step)
            opt_critic.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm_param)
            opt_critic.step()
            lr_scheduler_critic.step()

            if (global_training_step + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(t + 1),
                                       'a_lr: {}'.format(lr_scheduler_actor.get_last_lr()[0]),
                                       'a_loss: {}'.format(action_loss),
                                       'c_lr: {}'.format(lr_scheduler_critic.get_last_lr()[0]),
                                       'c_loss: {}'.format(value_loss),
                                       'time: {:.4f}s'.format(time.time() - tt)
                                       ]))

            global_training_step += 1

        print('avg model mse loss:', np.mean(losses))
        # torch.save(model_G2.state_dict(), args.log_dir+'/param/model_G2_' + str(ID) + '_' + str(ep+1) + '.pkl')
        # torch.save(model_G5.state_dict(), args.log_dir+'/param/model_G5_' + str(ID) + '_' + str(ep+1) + '.pkl')
        # torch.save(model_G10.state_dict(), args.log_dir+'/param/model_G10_' + str(ID) + '_' + str(ep+1) + '.pkl')
        # torch.save(model_r.state_dict(), args.log_dir+'/param/model_r_' + str(ID) + '_' + str(ep+1) + '.pkl')
        # torch.save(actor.state_dict(), args.log_dir+'/param/model_actor_' + str(ID) + '_' + str(ep+1) + '.pkl')
        # torch.save(critic.state_dict(), args.log_dir+'/param/model_critic_' + str(ID) + '_' + str(ep+1) + '.pkl')

    print('avg model mse loss:', np.mean(losses_list))
