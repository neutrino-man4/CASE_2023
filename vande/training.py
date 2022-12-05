import os
import time
import datetime
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import vande.vae.losses as losses
from loguru import logger

#logger.add(f"{experiment.model_dir}/logfile.txt")

class Stopper():
    def __init__(self, optimizer, min_delta=0.01, patience=4, max_lr_decay=10, lr_decay_factor=0.3):
        self.optimizer = optimizer
        self.min_delta = min_delta
        self.patience = patience
        self.patience_curr = 0
        self.max_lr_decay = max_lr_decay
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_n = 0

    def callback_early_stopping(self, loss_list, min_delta=0.1, patience=4):
        # increase loss-window width at each epoch
        self.patience_curr += 1
        if self.patience_curr < patience:
            return False
        # compute difference of the last #patience epoch losses
        mean = np.mean(loss_list[-patience:])
        deltas = np.absolute(np.diff(loss_list[-patience:])) 
        # return true if all relative deltas are smaller than min_delta
        return np.all((deltas / mean) < min_delta)

    def check_stop_training(self, losses):
        if self.callback_early_stopping(losses, min_delta=self.min_delta, patience=self.patience):
            print('-'*7 + ' Early stopping for last '+ str(self.patience)+ ' validation losses ' + str([l.numpy() for l in losses[-self.patience:]]) + '-'*7)
            if self.lr_decay_n >= self.max_lr_decay:
                # stop the training
                return True
            else: 
                # decrease the learning rate
                curr_lr = self.optimizer.learning_rate.numpy()
                self.optimizer.learning_rate.assign(curr_lr * self.lr_decay_factor)
                self.lr_decay_n += 1
                # reset patience window each time lr has been decreased
                self.patience_curr = 0
                print('decreasing learning rate from {:.3e} to {:.3e}'.format(curr_lr, self.optimizer.learning_rate.numpy()))
        return False

class Trainer():

    def __init__(self, optimizer, beta=0.1, patience=4, min_delta=0.01, max_lr_decay=4, lambda_reg=0.0,annealing=False,datalength=2e6,batchsize=64,lr_decay_factor=0.3):
        self.optimizer = optimizer
        self.beta = beta
        self.patience = patience
        self.min_delta = min_delta
        self.max_lr_decay = max_lr_decay
        self.lambda_reg = lambda_reg
        self.annealing = annealing
        self.datalength = datalength
        self.batchsize = batchsize
        self.lr_decay_factor = lr_decay_factor
        self.train_stop = Stopper(optimizer, min_delta, patience, max_lr_decay,lr_decay_factor)
        self.best_loss_so_far = None

    def check_best_model(self, valid_loss):
        if (self.best_loss_so_far is None) or (valid_loss < self.best_loss_so_far):
            self.best_loss_so_far = valid_loss
            return True
        return False 

    @tf.function
    def training_step(self, model, loss_fn_reco, x_batch, beta):
        
        with tf.GradientTape() as tape:
            # Run the forward pass
            predictions = model(x_batch, training=True)  # Logits for this minibatch
            # Compute the loss value for this minibatch.
            #print(x_batch)
            reco_loss = tf.math.reduce_mean(loss_fn_reco(x_batch, predictions))
            kl_loss = tf.math.reduce_mean(model.losses) # get kl loss registered in sampling layer
            reg_loss = losses.l2_regularize(model.trainable_weights)
            #total_loss = reco_loss + self.beta * kl_loss + self.lambda_reg * reg_loss
            total_loss = reco_loss + beta * kl_loss + self.lambda_reg * reg_loss
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(total_loss, model.trainable_weights)
        # Run one step of gradient descent
        self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return reco_loss, beta*kl_loss, reg_loss


    def training_epoch(self, model, loss_fn, train_ds):
        
        # metric (reset at start of each epoch)
        training_loss_reco = 0.
        training_loss_kl = 0.

        beta = 0
        nsteps_in_dataset = int(self.datalength/self.batchsize)
        # for each batch in training set
        for step, x_batch_train in enumerate(train_ds):
            
            if self.annealing:
                if step < nsteps_in_dataset/2:
                    beta += 2*self.beta/nsteps_in_dataset
                    #beta = 0.
                else:
                    beta = self.beta
                beta = tf.convert_to_tensor(beta)
                #print(beta)
                reco_loss, kl_loss, reg_loss = self.training_step(model, loss_fn, x_batch_train, beta)
            else:

                reco_loss, kl_loss, reg_loss = self.training_step(model, loss_fn, x_batch_train, 0.5)

            # add training loss (reco & KL)
            training_loss_reco += reco_loss
            training_loss_kl += kl_loss
            
            # Log every 4000 batches.
            if step % 4000 == 0:
                print("Step {}: mean reco loss {:.4f}, KL loss {:.4f}, Regul loss {:.4f} (in one batch)".format(step, float(reco_loss), float(kl_loss), float(reg_loss)))
                print("Seen so far: %s samples" % ((step + 1) * self.batchsize))

        # return average batch loss
        return (training_loss_reco / (step+1), training_loss_kl / (step+1)) 


    @tf.function
    def validation_step(self, model, loss_fn, x_batch):
        predictions = model(x_batch, training=False)
        reco_loss = tf.math.reduce_mean(loss_fn(x_batch, predictions))
        kl_loss = tf.math.reduce_mean(model.losses)
        return reco_loss, self.beta * kl_loss


    def validation_epoch(self, model, loss_fn, valid_ds):
        
        validation_loss_reco = 0.
        validation_loss_kl = 0.

        for step, x_batch_val in enumerate(valid_ds):
            reco_loss, kl_loss = self.validation_step(model, loss_fn, x_batch_val[:,:,:3])
            validation_loss_reco += reco_loss
            validation_loss_kl += kl_loss

        return (validation_loss_reco / (step+1), validation_loss_kl / (step+1))


    def train(self, vae, loss_fn, train_ds, valid_ds, epochs, model_dir):

        print("Saving model to %s"%os.path.join(model_dir))
        logger.add("%s/training.log"%os.path.join(model_dir),level='DEBUG')
        logger.debug("Saving model to %s"%os.path.join(model_dir))
        
        model = vae.model

        losses_reco = []
        losses_valid = []

        for epoch in range(epochs):
            now = datetime.datetime.today()
            print("\n### [{}.{} {}:{}:{}] Start of epoch {}".format(now.day, now.month, now.hour, now.minute, now.second, epoch))
            start_time = time.time()
            training_loss_reco, training_loss_kl = self.training_epoch(model, loss_fn, train_ds)
            validation_loss_reco, validation_loss_kl = self.validation_epoch(model, loss_fn, valid_ds)
            losses_reco.append(training_loss_reco + training_loss_kl)
            losses_valid.append(validation_loss_reco + validation_loss_kl)    
            # print epoch results
            logger.debug('### [Epoch {} - {:.2f} sec]: train loss reco {:.3f} kl {:.3f}, val loss reco {:.3f} kl {:.3f} (mean / batch) ###'.format(epoch, time.time()-start_time, training_loss_reco, training_loss_kl, validation_loss_reco, validation_loss_kl))
            print('### [Epoch {} - {:.2f} sec]: train loss reco {:.3f} kl {:.3f}, val loss reco {:.3f} kl {:.3f} (mean / batch) ###'.format(epoch, time.time()-start_time, training_loss_reco, training_loss_kl, validation_loss_reco, validation_loss_kl))
            if self.train_stop.check_stop_training(losses_valid):
                print('!!! stopping training !!!')
                break
            if epoch > 4 and self.check_best_model(validation_loss_reco):
                print('saving best so far model with valid loss {:.3f} and kl loss {:.3f}'.format(validation_loss_reco, validation_loss_kl))
                vae.save(os.path.join(model_dir, 'best_so_far'))
        return losses_reco, losses_valid


# plot training results
def plot_training_results(losses_train, losses_valid, fig_dir, plot_suffix=''):
    plt.figure()
    plt.semilogy(losses_train)
    plt.semilogy(losses_valid)
    plt.title('training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training','validation'], loc='upper right')
    plt.savefig(os.path.join(fig_dir,'loss'+plot_suffix+'.png'))
    plt.close()

### routine for prediction

@tf.function
def prediction_step(model, loss_fn, x_batch):
    #print("WTF")
    #print(x_batch.shape)
    predictions = model(x_batch, training=False)
    #print("WTF2")
    #print(predictions)
    reco_loss = loss_fn(x_batch, predictions)
    #print("WTF3")
    #print(reco_loss)
    kl_loss = sum(model.losses)
    return predictions, reco_loss, kl_loss

@tf.function
def prediction_step_with_latent(model, loss_fn, x_batch):
    #print("WTF with latent")
    #print(x_batch.shape)
    #print(type(model))
    #print(model)
    predictions  = model(x_batch)
    return predictions[0], predictions[1], predictions[2]


def predict(model, loss_fn, test_ds):

    predictions = []
    originals = []
    validation_loss_reco = []
    validation_loss_kl = []

    for step, x_batch_test in enumerate(test_ds):
        # print('predicting batch {}'.format(step))

        #print("Model")
        #print(model)
        #print("X")
        #print(x_batch_test)


        predictions_batch, reco_loss, kl_loss = prediction_step(model, loss_fn, x_batch_test)
        validation_loss_reco.append(reco_loss)
        validation_loss_kl.append(kl_loss)

        #print("Batch")
        #print(predictions_batch)
        #tmp_predictions = np.argsort(np.asarray(predictions_batch)[...,0], axis=0)


        predictions.append(predictions_batch)
        #predictions.append(tmp_predictions)
        originals.append(x_batch_test)

    # import ipdb; ipdb.set_trace()    

    return (np.concatenate(predictions, axis=0), np.concatenate(validation_loss_reco, axis=0), np.concatenate(validation_loss_kl, axis=0), np.concatenate(originals, axis=0))




def predict_with_latent(model, loss_fn, test_ds):

    z1s = []
    z2s = []
    z3s = []

    for step, x_batch_test in enumerate(test_ds):
        # print('predicting batch {}'.format(step))

        #print("Model")
        #print(model)
        #print("X")
        #print(x_batch_test)


        z_mean, z_log_var, zs = prediction_step_with_latent(model, loss_fn, x_batch_test)

        z1s.append(z_mean)
        z2s.append(z_log_var)
        z3s.append(zs)
        #predictions.append(tmp_predictions)
        #zs.append(z)

    # import ipdb; ipdb.set_trace()    

    
    return (z1s,z2s,z3s)
