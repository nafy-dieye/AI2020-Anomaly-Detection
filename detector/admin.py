from django.contrib import admin

from .models import SensorReading


@admin.register(SensorReading)
class SensorReadingAdmin(admin.ModelAdmin):
    list_display = (
        "timestamp",
        "is_anomaly",
        "reconstruction_error",
        "threshold",
        "severity_ratio",
    )
    list_filter = ("is_anomaly",)
    ordering = ("-timestamp",)
