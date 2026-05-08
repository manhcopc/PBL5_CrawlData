import logging
import requests
from typing import Dict, Any

# Configure logging for the webhook
logger = logging.getLogger("webhook_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [GEN SERVICE] [WEBHOOK] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def send_webhook_callback(callback_url: str, payload: Dict[str, Any], timeout: int = 10) -> bool:
    """
    Sends an asynchronous webhook callback to the specified URL with robust error handling.
    
    Args:
        callback_url (str): The URL to send the webhook to.
        payload (Dict[str, Any]): The JSON payload to send in the webhook body.
        timeout (int): The timeout for the request in seconds. Defaults to 10.
        
    Returns:
        bool: True if the webhook was sent successfully, False otherwise.
    """
    try:
        logger.info(f"Sending webhook to {callback_url}")
        response = requests.post(callback_url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"Successfully delivered callback to {callback_url} (Status: {response.status_code})")
        return True
    except requests.exceptions.Timeout as exc:
        logger.error(f"Timeout while sending webhook to {callback_url}: {exc}")
    except requests.exceptions.ConnectionError as exc:
        logger.error(f"Connection error while sending webhook to {callback_url}: {exc}")
    except requests.exceptions.HTTPError as exc:
        logger.error(f"HTTP error for webhook {callback_url}: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error while sending webhook to {callback_url}: {exc}")
    
    return False
