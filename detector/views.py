import json
from datetime import timedelta

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from .forms import SensorReadingForm
from .ml_model import RAW_INPUT_FIELDS, AnomalyModel, ANNModel
from .models import SensorReading


def _save_reading(raw: dict, result: dict) -> SensorReading:
    return SensorReading.objects.create(
        model_used=result["model_used"],
        air_temperature=raw["air_temperature"],
        process_temperature=raw["process_temperature"],
        rotational_speed=raw["rotational_speed"],
        torque=raw["torque"],
        tool_wear=raw["tool_wear"],
        reconstruction_error=result["score"],
        threshold=result["threshold"],
        is_anomaly=result["is_anomaly"],
        severity_ratio=result["severity_ratio"],
    )


def index(request):
    result = None
    reading_id = None

    selected_model = (
        request.POST.get("model_choice", "autoencoder")
        if request.method == "POST"
        else "autoencoder"
    )

    if selected_model not in {"autoencoder", "ann"}:
        selected_model = "autoencoder"

    current_model = (
        ANNModel.get_instance()
        if selected_model == "ann"
        else AnomalyModel.get_instance()
    )

    if request.method == "POST":
        form = SensorReadingForm(request.POST)

        if form.is_valid():
            raw = {
                field: form.cleaned_data[field]
                for field in RAW_INPUT_FIELDS
            }

            machine_type = (
                form.cleaned_data.get("machine_type") or "M"
            )

            if selected_model == "ann":
                result = ANNModel.get_instance().predict(
                    raw,
                    machine_type,
                )
            else:
                result = AnomalyModel.get_instance().predict(raw)

            reading = _save_reading(raw, result)
            reading_id = reading.id
    else:
        form = SensorReadingForm()

    gauge_position = None

    if result:
        capped_ratio = min(
            result["severity_ratio"] or 0,
            1.5,
        )
        gauge_position = round(
            capped_ratio / 1.5 * 100,
            1,
        )

    threshold_position = round(
        min(current_model.threshold / 1.5, 1.0) * 100,
        1,
    )

    context = {
        "form": form,
        "result": result,
        "reading_id": reading_id,
        "gauge_position": gauge_position,
        "gauge_threshold_position": threshold_position,
        "current_threshold": current_model.threshold,
        "current_threshold_percentile": getattr(
            current_model,
            "threshold_percentile",
            None,
        ),
        "selected_model": selected_model,
    }

    return render(
        request,
        "detector/index.html",
        context,
    )


def flag_reading(request, pk):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Méthode non autorisée."},
            status=405,
        )

    try:
        reading = SensorReading.objects.get(pk=pk)
    except SensorReading.DoesNotExist:
        return JsonResponse(
            {"error": "Lecture introuvable."},
            status=404,
        )

    reading.flagged = True
    reading.flag_comment = request.POST.get(
        "comment",
        "",
    )[:280]

    reading.save(
        update_fields=[
            "flagged",
            "flag_comment",
        ]
    )

    next_url = request.POST.get(
        "next"
    ) or reverse("detector:index")

    return redirect(next_url)


def reset_history(request):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Méthode non autorisée."},
            status=405,
        )

    deleted_count, _ = SensorReading.objects.all().delete()

    return redirect(
        reverse("detector:dashboard")
        + f"?reset={deleted_count}"
    )


def dashboard(request):
    readings_qs = SensorReading.objects.all()

    total_count = readings_qs.count()
    anomaly_count = readings_qs.filter(
        is_anomaly=True
    ).count()

    anomaly_rate = (
        anomaly_count / total_count * 100
        if total_count
        else 0
    )

    last_anomaly = readings_qs.filter(
        is_anomaly=True
    ).first()

    last_24h_cutoff = (
        timezone.now()
        - timedelta(hours=24)
    )

    anomalies_last_24h = readings_qs.filter(
        is_anomaly=True,
        timestamp__gte=last_24h_cutoff,
    ).count()

    failure_log = list(
        readings_qs.filter(
            is_anomaly=True
        ).values(
            "id",
            "timestamp",
            "model_used",
            "reconstruction_error",
            "threshold",
            "severity_ratio",
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
            "flagged",
        )[:50]
    )

    flagged_count = readings_qs.filter(
        flagged=True
    ).count()

    chart_readings = list(
        readings_qs.order_by(
            "-timestamp"
        ).values(
            "timestamp",
            "reconstruction_error",
            "is_anomaly",
            "model_used",
        )[:200]
    )

    chart_readings.reverse()

    chart_data = {
        "labels": [
            r["timestamp"].strftime(
                "%d/%m %H:%M:%S"
            )
            for r in chart_readings
        ],
        "errors": [
            r["reconstruction_error"]
            for r in chart_readings
        ],
        "is_anomaly": [
            r["is_anomaly"]
            for r in chart_readings
        ],
        "models": [
            r["model_used"]
            for r in chart_readings
        ],
    }

    threshold_value = (
        readings_qs.first().threshold
        if total_count
        else None
    )

    context = {
        "total_count": total_count,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "last_anomaly": last_anomaly,
        "anomalies_last_24h": anomalies_last_24h,
        "failure_log": failure_log,
        "flagged_count": flagged_count,
        "chart_data_json": json.dumps(
            chart_data
        ),
        "threshold_value": threshold_value,
    }

    return render(
        request,
        "detector/dashboard.html",
        context,
    )


@method_decorator(
    csrf_exempt,
    name="dispatch",
)
class PredictAPIView(View):
    # API conservée sur l'auto-encodeur
    # pour ne pas casser l'endpoint existant.

    def post(
        self,
        request,
        *args,
        **kwargs,
    ):
        try:
            payload = json.loads(
                request.body.decode(
                    "utf-8"
                )
            )
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
        ):
            return JsonResponse(
                {
                    "error": (
                        "Corps de requête JSON invalide."
                    )
                },
                status=400,
            )

        missing = [
            field
            for field in RAW_INPUT_FIELDS
            if field not in payload
        ]

        if missing:
            return JsonResponse(
                {
                    "error": (
                        "Champs manquants : "
                        + ", ".join(missing)
                    )
                },
                status=400,
            )

        try:
            raw = {
                field: float(payload[field])
                for field in RAW_INPUT_FIELDS
            }
        except (
            TypeError,
            ValueError,
        ):
            return JsonResponse(
                {
                    "error": (
                        "Toutes les valeurs "
                        "doivent être numériques."
                    )
                },
                status=400,
            )

        if (
            raw["air_temperature"] == 0
            or raw["rotational_speed"] == 0
        ):
            return JsonResponse(
                {
                    "error": (
                        "air_temperature et "
                        "rotational_speed ne peuvent "
                        "pas être nuls."
                    )
                },
                status=400,
            )

        result = AnomalyModel.get_instance().predict(
            raw
        )

        _save_reading(
            raw,
            result,
        )

        return JsonResponse(result)

    def get(
        self,
        request,
        *args,
        **kwargs,
    ):
        return JsonResponse(
            {
                "detail": (
                    "Utilisez POST avec un corps JSON "
                    "pour obtenir une prédiction."
                ),
                "expected_fields": RAW_INPUT_FIELDS,
            }
        )
