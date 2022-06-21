import torch
import numpy as np
import pickle
from WGAN_GRUI import PretrainGenerator, Generator, Discriminator
from loss import MRLoss, DLoss, GLoss, ImpLoss

torch.manual_seed(0)

def shuffle(x, y, z):
    # Turn into numpy
    if torch.is_tensor(x) == True:
        x = x.numpy()
    if torch.is_tensor(y) == True:
        y = y.numpy()
    if torch.is_tensor(z) == True:
        z = z.numpy()
    # Randomly exchange
    for i in range(len(x)):
        j = int(np.random.random() * (i + 1))
        if j <= len(x)-1:
            x[i], x[j] = x[j], x[i]
            y[i], y[j] = y[j], y[i]
            z[i], z[j] = z[j], z[i]

    # Turn back to tensor
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    z = torch.from_numpy(z)
    return x, y, z


def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def pretrain(model, X, M, Delta, batch_size, num_epoch, lr):
    criterion = MRLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)
    for epoch in range(num_epoch):
        X, M, Delta = shuffle(X, M, Delta)
        for step in range(X.shape[0] // batch_size + 1):
            if (step + 1) * batch_size <= X.shape[0]:
                X_batch = X[int(step * batch_size):int((step + 1) * batch_size)]
                M_batch = M[int(step * batch_size):int((step + 1) * batch_size)]
                Delta_batch = Delta[int(step * batch_size):int((step + 1) * batch_size)]
            else:
                X_batch = X[int(step * batch_size):]
                M_batch = M[int(step * batch_size):]
                Delta_batch = Delta[int(step * batch_size):]
            outputs = model(X_batch.transpose(0, 1), Delta_batch.transpose(0, 1), None)
            X_batch_g = outputs.transpose(0, 1)
            loss = criterion(X_batch, M_batch, X_batch_g)
            optimizer.zero_grad()
            loss.backward()
            grad_clipping(model, 0.99)
            optimizer.step()
            # display training status
            print('Epoch: [%2d] [%4d/%4d] pretrain_loss: %.8f' % (epoch+1, step+1, X.shape[0] // batch_size + 1, loss))
    return model.state_dict()


def train(X, M, Delta, batch_size, num_epoch, num_pretrain_epoch, lr, disc_iters):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_model = PretrainGenerator(X.shape[2], 64, 60, 0.5).to(device)
    generator = Generator(X.shape[2], 64, 60, 64, 0.5).to(device)
    discriminator = Discriminator(X.shape[2], 64, 0.5).to(device)
    
    X = torch.from_numpy(X).float()
    M = torch.from_numpy(M).float()
    Delta = torch.from_numpy(Delta).float()
    
    pretrained_dict = pretrain(pretrain_model, X, M, Delta, batch_size, num_pretrain_epoch, lr * 2)
    generator_dict = generator.state_dict()
    generator_dict.update(pretrained_dict)
    generator.load_state_dict(generator_dict)
    
    d_criterion = DLoss()
    g_criterion = GLoss()
    d_optimizer = torch.optim.SGD(discriminator.parameters(), lr)
    g_optimizer = torch.optim.SGD(generator.parameters(), lr * disc_iters)
    
    counter = 1
    
    for epoch in range(num_pretrain_epoch + 1, num_epoch+1):
        X, M, Delta = shuffle(X, M, Delta)
        for step in range(X.shape[0] // batch_size + 1):
            if (step + 1) * batch_size <= X.shape[0]:
                X_batch = X[int(step * batch_size):int((step + 1) * batch_size)]
                # M_batch = M[int(step * batch_size):int((step + 1) * batch_size)]
                Delta_batch = Delta[int(step * batch_size):int((step + 1) * batch_size)]
            else:
                X_batch = X[int(step * batch_size):]
                # M_batch = M[int(step * batch_size):]
                Delta_batch = Delta[int(step * batch_size):]
            
            # Update Discriminator Networks
            outputs, delta, final_state = generator(None, None, batch_size)
            d_real_probs, d_real_logits = discriminator(X_batch.transpose(0, 1), Delta_batch.transpose(0, 1), None)
            d_fake_probs, d_fake_logits = discriminator(outputs, delta, None)
            d_loss = d_criterion(d_real_logits, d_fake_logits)
            d_optimizer.zero_grad()
            d_loss.backward()
            grad_clipping(discriminator, 0.99)
            d_optimizer.step()
            print('Epoch: [%2d] [%4d/%4d] d_loss: %.8f, counter: %4d'% (epoch, step+1, X.shape[0] // batch_size + 1, d_loss, counter))
            
            # Update Generator Networks
            if counter % disc_iters == 0:
                outputs, delta, final_state = generator(None, None, batch_size)
                d_real_probs, d_real_logits = discriminator(X_batch.transpose(0, 1), Delta_batch.transpose(0, 1), None)
                d_fake_probs, d_fake_logits = discriminator(outputs, delta, None)
                d_loss = d_criterion(d_real_logits, d_fake_logits)
                g_loss = g_criterion(d_fake_logits)
                g_optimizer.zero_grad()
                g_loss.backward()
                grad_clipping(generator, 0.99)
                g_optimizer.step()
                print('Epoch: [%2d] [%4d/%4d] d_loss: %.8f, g_loss: %.8f, counter: %4d'\
                      % (epoch, step+1, X.shape[0] // batch_size + 1, d_loss, g_loss, counter))
            
            counter += 1
    
    return generator, discriminator


def imputation(generator, discriminator, X, M, z_dim, batch_size, lr, g_loss_lambda, impute_iters):
    batch_id = 1
    impute_times = 0
    counter = 0
    
    X = torch.from_numpy(X).float()
    M = torch.from_numpy(M).float()
    
    X_imputed = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for step in range(X.shape[0] // batch_size + 1):
        if (step + 1) * batch_size <= X.shape[0]:
            X_batch = X[int(step * batch_size):int((step + 1) * batch_size)]
            M_batch = M[int(step * batch_size):int((step + 1) * batch_size)]
        else:
            X_batch = X[int(step * batch_size):]
            M_batch = M[int(step * batch_size):]
        
        generator_dict = generator.state_dict()
        gen = Generator(X.shape[2], 64, 60, 64, 0).to(device)
        gen_dict = gen.state_dict()
        gen_dict.update(generator_dict)
        gen.load_state_dict(gen_dict)

        criterion = ImpLoss(g_loss_lambda)
        optimizer = torch.optim.SGD(gen.parameters(), lr * 7)
        
        Z = torch.randn((X.shape[1], batch_size, z_dim)) * 0.1
        init_state = None
        for i in range(impute_iters):
            outputs, delta, final_state = gen(Z, init_state, batch_size)
            init_state = final_state.detach()
            imputed_fake_probs, imputed_fake_logits = discriminator(outputs, delta, None)
            loss = criterion(X_batch, M_batch, outputs, imputed_fake_logits)
            optimizer.zero_grad()
            loss.backward()
            grad_clipping(generator, 0.99)
            optimizer.step()
            impute_times += 1
            counter += 1
            if counter % 5 == 0:
                print("Batch ID: [%2d] [%2d/%2d] Imputation Loss: %.8f" % (batch_id, impute_times, impute_iters, loss))
        for j in range(X_batch.shape[0]):
            X_imputed.append((X_batch[j] * M_batch[j] + outputs[:,j,:] * (1 - M_batch[j])).detach().numpy())
        batch_id += 1
        impute_times = 0
    
    with open('E:/WashU/Research/ICU/Data/val/X_val_sliced_norm_nearestGAN.pkl', 'wb') as f:
        pickle.dump(X_imputed, f)
        f.close()