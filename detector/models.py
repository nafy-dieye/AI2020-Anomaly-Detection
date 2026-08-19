from django.db import models


class SensorReading(models.Model):
    """
    Historique des lectures capteurs analysées par le modèle, pour alimenter
    le dashboard (statistiques, journal des pannes, graphique dans le temps).
    """

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    # Variables brutes saisies par l'utilisateur
    air_temperature = models.FloatField(verbose_name="Température de l'air (K)")
    process_temperature = models.FloatField(verbose_name="Température du process (K)")
    rotational_speed = models.FloatField(verbose_name="Vitesse de rotation (rpm)")
    torque = models.FloatField(verbose_name="Couple (Nm)")
    tool_wear = models.FloatField(verbose_name="Usure de l'outil (min)")

    # Modèle ayant produit le verdict
    model_used = models.CharField(
        max_length=20,
        default="autoencoder",
        choices=[
            ("autoencoder", "Auto-encodeur"),
            ("ann", "ANN"),
        ],
        db_index=True,
    )

    # Score du modèle :
    # erreur de reconstruction pour l'auto-encodeur,
    # score sigmoïde pour l'ANN.
    reconstruction_error = models.FloatField()
    threshold = models.FloatField()
    is_anomaly = models.BooleanField(db_index=True)
    severity_ratio = models.FloatField(null=True, blank=True)

    # Signalement manuel (ex : un opérateur juge que le verdict du modèle est faux)
    flagged = models.BooleanField(default=False, db_index=True, verbose_name="Signalée comme incorrecte")
    flag_comment = models.CharField(max_length=280, blank=True, default="")

    class Meta:
        ordering = ["-timestamp"]
        verbose_name = "Lecture capteur"
        verbose_name_plural = "Lectures capteurs"

    def __str__(self):
        status = "ANOMALIE" if self.is_anomaly else "normal"
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {status} (err={self.reconstruction_error:.3f})"
