from CoreModels import dcgan, wgan, infogan
#vs2626
class repository(object):
    """Repository which gives generators / discriminators based on gan_type"""
    @staticmethod
    def generator_factory(type):
        """Repository which gives generators based on gan_type"""
        if type == 'dcgan':
            return dcgan.generator()
        elif type == 'wgan':
            return wgan.generator()
        elif type == 'infogan':
            return infogan.generator()
        

    @staticmethod
    def discriminator_factory(type):
        """Repository which gives discriminators based on gan_type"""
        if type == 'dcgan':
            return dcgan.discriminator()
        elif type == 'wgan':
            return wgan.discriminator()
        elif type == 'infogan':
            return infogan.discriminator()