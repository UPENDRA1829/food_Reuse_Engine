import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import settings
from .service import run_daily_batch, run_retrain

log = logging.getLogger(__name__)
_sched: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler:
    global _sched
    _sched = BackgroundScheduler()
    _sched.add_job(run_daily_batch, CronTrigger.from_crontab(settings.batch_cron),
                   id="daily_batch", max_instances=1, coalesce=True)
    _sched.add_job(run_retrain, CronTrigger.from_crontab(settings.retrain_cron),
                   id="weekly_retrain", max_instances=1, coalesce=True)
    _sched.start()
    log.info("scheduler started: batch=%s retrain=%s", settings.batch_cron, settings.retrain_cron)
    return _sched


def stop_scheduler():
    if _sched:
        _sched.shutdown(wait=False)
