import numpy as np 
from helper import VisdomPlotter

class Logger(object):
    def __init__(self, vis_screen):
        self.plotter = VisdomPlotter(env_name=vis_screen)
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def draw(self, CorrectImages, Unrealimages):
        self.plotter.draw('generated images', Unrealimages.data.cpu().numpy()[:64] * 128 + 128)
        self.plotter.draw('real images', CorrectImages.data.cpu().numpy()[:64] * 128 + 128)

    def plotter_epoch(self, epoch):
        self.plotter.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        self.plotter.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        self.hist_D = []
        self.hist_G = []

    def plotter_epoch_w(self, epoch):
        self.plotter.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        self.plotter.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        self.plotter.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).mean())
        self.plotter.plot('D(G(X))', 'train', epoch, np.array(self.hist_DGx).mean())
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def logger_iter_gan(self, epoch, d_loss, g_loss, real_score, Unrealscore):
        print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
            epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
            Unrealscore.data.cpu().mean()))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())
        self.hist_Dx.append(real_score.data.cpu().mean())
        self.hist_DGx.append(Unrealscore.data.cpu().mean())

    def logger_iter_wgan(self, epoch, gen_iteration, d_loss, g_loss, real_loss, Unrealloss):
        print("Epoch: %d, Gen_iteration: %d, d_loss= %f, g_loss= %f, real_loss= %f, Unrealloss = %f" %
              (epoch, gen_iteration, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_loss, Unrealloss))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())

    def logger_iter_infogan(self, epoch, d_loss, g_loss, real_score, Unrealscore):
        print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
            epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
            Unrealscore.data.cpu().mean()))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())
        self.hist_Dx.append(real_score.data.cpu().mean())
        self.hist_DGx.append(Unrealscore.data.cpu().mean())
