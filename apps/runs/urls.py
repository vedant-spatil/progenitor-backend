from django.urls import path

from apps.runs import browse, demo, views

urlpatterns = [
    path("me", views.MeView.as_view(), name="me"),
    path("overview", browse.OverviewView.as_view(), name="overview"),
    path("patients", views.PatientListView.as_view(), name="patient-list"),
    path("patients/<int:patient_id>", browse.PatientDetailView.as_view(), name="patient-detail"),
    path("trials", browse.TrialListView.as_view(), name="trial-list"),
    path("trials/<str:nct_id>", browse.TrialDetailView.as_view(), name="trial-detail"),
    path("referrals", browse.ReferralInboxView.as_view(), name="referral-list"),
    path("referrals/<int:referral_id>", views.ReferralDetailView.as_view(), name="referral-detail"),
    path("audit", browse.AuditLogView.as_view(), name="audit-log"),
    path("evals", browse.EvalReportsView.as_view(), name="eval-reports"),
    path("demo", demo.DemoListView.as_view(), name="demo-list"),
    path("demo/reset", demo.DemoResetView.as_view(), name="demo-reset"),
    path("demo/<int:patient_id>/start", demo.DemoStartView.as_view(), name="demo-start"),
    path("runs", views.RunCreateView.as_view(), name="run-create"),
    path("runs/<uuid:run_id>", views.RunDetailView.as_view(), name="run-detail"),
    path("runs/<uuid:run_id>/events", views.run_events, name="run-events"),
    path("runs/<uuid:run_id>/shortlist", views.ShortlistView.as_view(), name="run-shortlist"),
    path("runs/<uuid:run_id>/answer", views.AnswerView.as_view(), name="run-answer"),
    path("runs/<uuid:run_id>/review", views.ReviewView.as_view(), name="run-review"),
    path("runs/<uuid:run_id>/referrals", views.ReferralsView.as_view(), name="run-referrals"),
]
