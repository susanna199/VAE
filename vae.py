
# --- 7. VAE MODEL CLASS ---

class VAE(keras.Model):
    """
    A VAE model that combines the encoder and decoder.
    Includes the custom training step with reconstruction and KL loss.
    """
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
    def get_config(self):
        config = super(VAE, self).get_config()
        # Note: do NOT include encoder and decoder in config directly
        return config

    @classmethod
    def from_config(cls, config):
        # encoder and decoder must be set manually after loading
        return cls(encoder=None, decoder=None, **config)
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Get latent space parameters and sampled vector z from the encoder
            z_mean, z_log_var, z = self.encoder(data)
            # Reconstruct the image using the decoder
            reconstruction = self.decoder(z)

            # --- Calculate Losses ---
            # 1. Reconstruction Loss (how well the VAE reconstructs the input)
            # We use binary cross-entropy, common for image reconstruction
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )

            # 2. KL Divergence Loss (regularization term)
            # This pushes the latent distribution to be close to a standard normal distribution
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Total loss is the sum of both
            total_loss = reconstruction_loss + kl_loss

        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update and return the loss metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
