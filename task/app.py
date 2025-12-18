import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import MASCoordinator
from task.logging_config import setup_logging, get_logger

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
UMS_AGENT_ENDPOINT = os.getenv('UMS_AGENT_ENDPOINT', "http://localhost:8042")
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

setup_logging(log_level=LOG_LEVEL)
logger = get_logger(__name__)


class MASCoordinatorApplication(ChatCompletion):

    async def chat_completion(self, request: Request, response: Response) -> None:
        conversation_id = request.headers.get('x-conversation-id', 'unknown')
        logger.info(f"Chat request is processing for conversation {conversation_id}")

        try:
            with response.create_single_choice() as choice:

                await MASCoordinator(
                    endpoint=DIAL_ENDPOINT,
                    deployment_name=DEPLOYMENT_NAME,
                    ums_agent_endpoint=UMS_AGENT_ENDPOINT
                ).handle_request(
                    choice=choice,
                    request=request,
                )
        except Exception as e:
            logger.error(
                f"Error during processing chat completion for conversation {conversation_id}: {str(e)}",
                exc_info=True
            )
            raise


def main():
    app = DIALApp()
    agent_app = MASCoordinatorApplication()
    app.add_chat_completion(deployment_name="mas-coordinator", impl=agent_app)

    uvicorn.run(app, port=8055, host="0.0.0.0")

if __name__ == "__main__":
    main()
