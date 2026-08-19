from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detector", "0003_alter_sensorreading_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="sensorreading",
            name="model_used",
            field=models.CharField(
                choices=[
                    ("autoencoder", "Auto-encodeur"),
                    ("ann", "ANN"),
                ],
                db_index=True,
                default="autoencoder",
                max_length=20,
            ),
        ),
    ]
