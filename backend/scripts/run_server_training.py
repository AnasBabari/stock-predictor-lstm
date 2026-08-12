import argparse
import logging
import sys
import time

from artifacts.signing import Ed25519ManifestSigner
from config import settings
from server_models.api import get_registry, get_storage
from server_models.retention import sweep_expired_bundles
from server_models.training import train_server_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run server-side background training.")
    parser.add_argument(
        "--once", action="store_true", help="Process the queue once and exit (cron mode)"
    )
    parser.add_argument(
        "--ticker", type=str, help="Train a specific ticker immediately (bypassing queue)"
    )
    parser.add_argument(
        "--gc-only", action="store_true", help="Prune expired bundle objects and exit"
    )
    args = parser.parse_args()

    # Check the signing key before constructing infrastructure clients so a
    # missing key surfaces as the intended clear exit code instead of failing
    # while connecting to Postgres/S3 first.
    if not settings.server_forecast_private_key_path:
        logger.error("server_forecast_private_key_path is not configured; cannot sign artifacts.")
        sys.exit(2)
    signer = Ed25519ManifestSigner.from_pem_file(settings.server_forecast_private_key_path)

    registry = get_registry()
    storage = get_storage()
    # The trainer owns the bucket: create it on first run (MinIO/dev) and make
    # the failure loud instead of letting every put_bundle race a missing bucket.
    storage.ensure_bucket()
    pruned = sweep_expired_bundles(
        registry,
        storage,
        retention_days=settings.server_bundle_retention_days,
    )
    logger.info(
        "Server bundle retention removed %d expired object(s) (window=%d days).",
        len(pruned),
        settings.server_bundle_retention_days,
    )
    if args.gc_only:
        sys.exit(0)

    if args.ticker:
        # Run explicitly for one ticker
        try:
            train_server_forecast(args.ticker, registry, storage, signer)
            logger.info(f"Successfully trained {args.ticker}")
        except Exception as e:
            logger.error(f"Failed to train {args.ticker}: {e}", exc_info=True)
            sys.exit(1)
        sys.exit(0)

    logger.info("Starting server training loop...")
    while True:
        try:
            job = registry.dequeue_job()
            if not job:
                if args.once:
                    logger.info("No more jobs. Exiting due to --once.")
                    break
                # Sleep and poll if not in --once mode
                time.sleep(5)
                continue

            ticker = job["ticker"]
            logger.info(f"Dequeued job {job['id']} for {ticker}")

            try:
                record = train_server_forecast(ticker, registry, storage, signer)
                if record:
                    registry.append_audit(
                        "training_success",
                        {
                            "ticker": ticker,
                            "version_id": record.key.version_id,
                            "job_id": job["id"],
                        },
                    )
                else:
                    registry.append_audit(
                        "training_failed",
                        {"ticker": ticker, "reason": "No candidate promoted", "job_id": job["id"]},
                    )
                registry.complete_job(job["id"])
            except Exception as e:
                logger.error(f"Failed to process job {job['id']}: {e}", exc_info=True)
                registry.append_audit(
                    "training_error", {"ticker": ticker, "error": str(e), "job_id": job["id"]}
                )
                registry.fail_job(job["id"], str(e))

        except Exception as e:
            logger.error(f"Error checking or processing queue: {e}", exc_info=True)
            if args.once:
                sys.exit(1)
            time.sleep(10)


if __name__ == "__main__":
    main()
