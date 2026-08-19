from django import forms


class SensorReadingForm(forms.Form):
    MODEL_CHOICES = [
        ("autoencoder", "Auto-encodeur"),
        ("ann", "ANN"),
    ]

    MACHINE_TYPE_CHOICES = [
        ("M", "M — moyen"),
        ("L", "L — faible"),
        ("H", "H — élevé"),
    ]

    model_choice = forms.ChoiceField(
        label="Modèle d'analyse",
        choices=MODEL_CHOICES,
        initial="autoencoder",
        widget=forms.RadioSelect,
    )

    machine_type = forms.ChoiceField(
        label="Type de machine (utilisé par l'ANN)",
        choices=MACHINE_TYPE_CHOICES,
        initial="M",
        required=False,
        widget=forms.Select,
    )

    air_temperature = forms.FloatField(
        label="Température de l'air (K)",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"step": "0.01", "placeholder": "ex : 298.1"}
        ),
    )

    process_temperature = forms.FloatField(
        label="Température du process (K)",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"step": "0.01", "placeholder": "ex : 308.6"}
        ),
    )

    rotational_speed = forms.FloatField(
        label="Vitesse de rotation (rpm)",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"step": "1", "placeholder": "ex : 1551"}
        ),
    )

    torque = forms.FloatField(
        label="Couple (Nm)",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"step": "0.1", "placeholder": "ex : 42.8"}
        ),
    )

    tool_wear = forms.FloatField(
        label="Usure de l'outil (min)",
        min_value=0,
        widget=forms.NumberInput(
            attrs={"step": "1", "placeholder": "ex : 108"}
        ),
    )

    def clean(self):
        cleaned_data = super().clean()

        air_temp = cleaned_data.get("air_temperature")
        process_temp = cleaned_data.get("process_temperature")
        rotational_speed = cleaned_data.get("rotational_speed")
        model_choice = cleaned_data.get("model_choice")
        machine_type = cleaned_data.get("machine_type")

        if air_temp is not None and air_temp == 0:
            self.add_error(
                "air_temperature",
                "La température de l'air ne peut pas être nulle.",
            )

        if rotational_speed is not None and rotational_speed == 0:
            self.add_error(
                "rotational_speed",
                "La vitesse de rotation ne peut pas être nulle.",
            )

        if air_temp is not None and process_temp is not None:
            if process_temp < air_temp:
                pass

        if model_choice == "ann" and not machine_type:
            self.add_error(
                "machine_type",
                "Choisissez le type de machine pour l'ANN.",
            )

        return cleaned_data
