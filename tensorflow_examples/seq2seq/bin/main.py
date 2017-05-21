import tensorflow as tf

tf.flags.DEFINE_boolean('train', False, 'run training if true, otherwise run inference')


def train():
    import bin.train
    tf.app.run(bin.train.main, ['', '--config_paths', 'config/model.yml,config/train.yml'])


def infer():
    import bin.infer
    tf.app.run(bin.infer.main, ['', '--config_path', 'config/infer.yml'])


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if tf.flags.FLAGS.train:
        train()
    else:
        infer()


if __name__ == '__main__':
    tf.app.run()
