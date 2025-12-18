import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName, CoordinationRequestWrapper
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        conversation_id = request.headers.get('x-conversation-id', 'unknown')

        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version='2025-01-01-preview'
        )

        logger.info(f"Calling coordinator for conversation {conversation_id}")
        coordination_stage = StageProcessor.open_stage(choice, "Coordination Request")
        coordination_request = await self.__prepare_coordination_request(
            client=client,
            request=request,
        )
        coordination_stage.append_content(f"```json\n\r{coordination_request.model_dump_json(indent=2)}\n\r```\n\r")
        StageProcessor.close_stage_safely(coordination_stage)

        agent_messages = []
        for agent_call in coordination_request.agent_calls:
            logger.info(f"Calling {agent_call.agent_name} agent for conversation {conversation_id}")
            processing_stage = StageProcessor.open_stage(choice, f"Call {agent_call.agent_name} Agent")
            agent_messages.append(await self.__handle_coordination_request(
                coordination_request=agent_call,
                choice=choice,
                stage=processing_stage,
                request=request,
            ))
            StageProcessor.close_stage_safely(processing_stage)

        logger.info(f"Generating final response for {conversation_id}")
        final_response = await self.__final_response(
            client=client,
            choice=choice,
            request=request,
            agent_messages=agent_messages,
        )

        logger.info(f"Final response for {conversation_id}: {final_response.json()}")

        return final_response

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequestWrapper:
        response = await client.chat.completions.create(
            messages=self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT),
            deployment_name=self.deployment_name,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequestWrapper.model_json_schema()
                    }
                },
            }
        )

        dict_content = json.loads(response.choices[0].message.content)
        return CoordinationRequestWrapper.model_validate(dict_content)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        messages = [
            {
                "role": Role.SYSTEM,
                "content": system_prompt,
            }
        ]
        for message in request.messages:
            if message.role == Role.USER and message.custom_content:
                messages.append(
                    {
                        "role": Role.USER,
                        "content": StrictStr(deepcopy(message.content)),
                    }
                )
            else:
                messages.append(message.dict(exclude_none=True))

        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        if coordination_request.agent_name is AgentName.GPA:
            return await GPAGateway(endpoint=self.endpoint).response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions,
            )
        elif coordination_request.agent_name is AgentName.UMS:
            return await UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint).response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=coordination_request.additional_instructions,
            )
        else:
            raise ValueError(f"Agent Name {coordination_request.agent_name} is unknown")

    async def __final_response(
            self, client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_messages: list[Message]
    ) -> Message:
        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)

        if len(agent_messages) > 1:
            context = "\n".join([f"## CONTEXT {i + 1}:\n {msg.content}" for i, msg in enumerate(agent_messages)])
        elif len(agent_messages) == 1:
            context = f"## CONTEXT:\n {agent_messages[0].content}"
        else:
            context = "No agent calls"
        updated_user_request = context + (
            "\n ---\n "
            f"## USER_REQUEST: \n {messages[-1]["content"]}"
        )
        messages[-1]["content"] = updated_user_request

        chunks = await client.chat.completions.create(
            messages=messages,
            deployment_name=self.deployment_name,
            stream=True
        )

        content = ''
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content

        custom_content = None
        for msg in agent_messages:
            if msg.custom_content:
                if custom_content is None:
                    custom_content = deepcopy(msg.custom_content)
                else:
                    custom_content.stages.extend(msg.custom_content.stages)
                    custom_content.attachments.extend(msg.custom_content.attachments)

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
            custom_content=custom_content
        )
