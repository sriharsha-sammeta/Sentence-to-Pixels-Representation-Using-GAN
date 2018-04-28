import numpy as np 
from helper import VisdomPlotter

#se2444

class Logger(object):
    """Used for logging the models outputs"""
    def __init__(self, vis_screen):
        self.plotter = VisdomPlotter(env_name=vis_screen)
        self.history_Disc = []
        self.history_Gen = []
        self.history_Discx = []
        self.history_DiscGx = []

    def draw(self, CorrectImages, Unrealimages):
        """ draw images: usually correct vs incorrect images in visdom server"""
        self.plotter.draw('generated images', Unrealimages.data.cpu().numpy()[:64] * 128 + 128)
        self.plotter.draw('real images', CorrectImages.data.cpu().numpy()[:64] * 128 + 128)

    def plotter_epoch(self, epoch):
        """Plots loss vs epochs"""
        self.plotter.plot('Discriminator', 'train', epoch, np.array(self.history_Disc).mean())
        self.plotter.plot('Generator', 'train', epoch, np.array(self.history_Gen).mean())
        self.history_Disc = []
        self.history_Gen = []

    def plotter_epoch_w(self, epoch):
        """Plots loss vs epochs and also D(X) and D(G(X)) """
        self.plotter.plot('Discriminator', 'train', epoch, np.array(self.history_Disc).mean())
        self.plotter.plot('Generator', 'train', epoch, np.array(self.history_Gen).mean())
        self.plotter.plot('D(X)', 'train', epoch, np.array(self.history_Discx).mean())
        self.plotter.plot('D(G(X))', 'train', epoch, np.array(self.history_DiscGx).mean())
        self.history_Disc = []
        self.history_Gen = []
        self.history_Discx = []
        self.history_DiscGx = []

    def logger_iter_gan(self, epoch, d_loss, g_loss, real_score, Unrealscore):
        """logger over iterations for dcgan which plots discrimintator loss, generator loss and D(X) and D(G(X))"""
        print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
            epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
            Unrealscore.data.cpu().mean()))
        self.history_Disc.append(d_loss.data.cpu().mean())
        self.history_Gen.append(g_loss.data.cpu().mean())
        self.history_Discx.append(real_score.data.cpu().mean())
        self.history_DiscGx.append(Unrealscore.data.cpu().mean())

    def logger_iter_wgan(self, epoch, gen_iteration, d_loss, g_loss, real_loss, Unrealloss):
        """logger over iterations for Wgan which plots discrimintator loss, generator loss and D(X) and D(G(X))"""
        print("Epoch: %d, Gen_iteration: %d, d_loss= %f, g_loss= %f, real_loss= %f, Unrealloss = %f" %
              (epoch, gen_iteration, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_loss, Unrealloss))
        self.history_Disc.append(d_loss.data.cpu().mean())
        self.history_Gen.append(g_loss.data.cpu().mean())

    def logger_iter_infogan(self, epoch, d_loss, g_loss, real_score, Unrealscore):
        """logger over iterations for infogan which plots discrimintator loss, generator loss and D(X) and D(G(X))"""
        print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
            epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
            Unrealscore.data.cpu().mean()))
        self.history_Disc.append(d_loss.data.cpu().mean())
        self.history_Gen.append(g_loss.data.cpu().mean())
        self.history_Discx.append(real_score.data.cpu().mean())
        self.history_DiscGx.append(Unrealscore.data.cpu().mean())
