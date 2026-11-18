from django.urls import path

from apps.runs import views

urlpatterns = [
    path("runs", views.RunCreateView.as_view(), name="run-create"),
    path("runs/<uuid:run_id>", views.RunDetailView.as_view(), name="run-detail"),
    path("runs/<uuid:run_id>/events", views.run_events, name="run-events"),
    path("runs/<uuid:run_id>/shortlist", views.ShortlistView.as_view(), name="run-shortlist"),
    path("runs/<uuid:run_id>/answer", views.AnswerView.as_view(), name="run-answer"),
    path("runs/<uuid:run_id>/review", views.ReviewView.as_view(), name="run-review"),
    path("runs/<uuid:run_id>/referrals", views.ReferralsView.as_view(), name="run-referrals"),
]
