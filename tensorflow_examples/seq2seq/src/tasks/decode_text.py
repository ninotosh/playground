import tensorflow as tf

from seq2seq.tasks import decode_text


class DecodeText(decode_text.DecodeText):
    def after_run(self, _run_context, run_values):
        super().after_run(_run_context, run_values)
        tf.logging.info('_run_context: {}, run_values: {}'.format(_run_context, run_values))
