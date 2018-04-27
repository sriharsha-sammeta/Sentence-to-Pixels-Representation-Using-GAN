from CoreModels import dcgan, wgan, infogan

class repository(object):

    @staticmethod
    def generator_factory(type):
        if type == 'dcgan':
            return dcgan.generator()
        elif type == 'wgan':
            return wgan.generator()
        elif type == 'infogan':
            return infogan.generator()
        

    @staticmethod
    def discriminator_factory(type):
        if type == 'dcgan':
            return dcgan.discriminator()
        elif type == 'wgan':
            return wgan.discriminator()
        elif type == 'infogan':
            return infogan.discriminator()