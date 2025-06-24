import json
import os
import time
import uuid
import hashlib
import threading
import requests
import base64
import re
from typing import Any, Dict, List, Optional, TypedDict, Union
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from cachetools import LRUCache


# Abacus Account Management
class AbacusAccount(TypedDict):
    _u_p: str
    _s_p: str
    is_valid: bool
    last_used: float
    error_count: int
    session_token: Optional[str]
    session_token_expires: float


# Global variables
VALID_CLIENT_KEYS: set = set()
ABACUS_ACCOUNTS: List[AbacusAccount] = []
ABACUS_MODELS: List[Dict[str, Any]] = []
account_rotation_lock = threading.Lock()
MAX_ERROR_COUNT = 3
ERROR_COOLDOWN = 300  # 5 minutes cooldown for accounts with errors
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

# LRU cache for conversation sessions (conversation_key -> (deploymentConversationId, account_index))
CONVERSATION_CACHE = LRUCache(maxsize=1000)
conversation_cache_lock = threading.Lock()


# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Any
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# FastAPI App
app = FastAPI(title="Abacus OpenAI API Adapter")
security = HTTPBearer(auto_error=False)


def log_debug(message: str):
    """Debug日志函数"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")


def load_client_api_keys():
    """Load client API keys from client_api_keys.json"""
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            VALID_CLIENT_KEYS = set(keys) if isinstance(keys, list) else set()
            print(f"Successfully loaded {len(VALID_CLIENT_KEYS)} client API keys.")
    except FileNotFoundError:
        print("Error: client_api_keys.json not found. Client authentication will fail.")
        VALID_CLIENT_KEYS = set()
    except Exception as e:
        print(f"Error loading client_api_keys.json: {e}")
        VALID_CLIENT_KEYS = set()


def load_abacus_accounts():
    """Load Abacus accounts from abacus.json"""
    global ABACUS_ACCOUNTS
    ABACUS_ACCOUNTS = []
    try:
        with open("abacus.json", "r", encoding="utf-8") as f:
            accounts = json.load(f)
            if not isinstance(accounts, list):
                print("Warning: abacus.json should contain a list of account objects.")
                return

            for acc in accounts:
                _u_p = acc.get("_u_p")
                _s_p = acc.get("_s_p")
                if _u_p and _s_p:
                    ABACUS_ACCOUNTS.append({
                        "_u_p": _u_p,
                        "_s_p": _s_p,
                        "is_valid": True,
                        "last_used": 0,
                        "error_count": 0,
                        "session_token": None,
                        "session_token_expires": 0
                    })
            print(f"Successfully loaded {len(ABACUS_ACCOUNTS)} Abacus accounts.")
    except FileNotFoundError:
        print("Error: abacus.json not found. API calls will fail.")
    except Exception as e:
        print(f"Error loading abacus.json: {e}")


def get_session_token(_u_p: str, _s_p: str) -> str:
    """Get session token for account"""
    url = "https://abacus.ai/api/v0/_getUserInfo"
    payload = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json",
        "Cookie": f'_u_p="{_u_p}"; _s_p="{_s_p}"',
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()
    return response.json()["result"]["sessionToken"]


def get_models_from_account(_u_p: str, _s_p: str) -> List[Dict[str, Any]]:
    """Get models list from account (for health check)"""
    url = "https://abacus.ai/api/v0/listExternalApplications"
    payload = {"includeSearchLlm": True, "isDesktop": True}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json",
        "Cookie": f'_u_p="{_u_p}"; _s_p="{_s_p}"',
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()
    return response.json()["result"]


def load_abacus_models():
    """Load Abacus models from first valid account"""
    global ABACUS_MODELS
    ABACUS_MODELS = []

    for account in ABACUS_ACCOUNTS:
        if not account["is_valid"]:
            continue

        try:
            models = get_models_from_account(account["_u_p"], account["_s_p"])
            # Filter visible models and add id field
            for model in models:
                if model.get("isVisible", False):
                    model["id"] = model["name"]  # Use name as id
                    model["owned_by"] = "abacus"
                    ABACUS_MODELS.append(model)

            print(f"Successfully loaded {len(ABACUS_MODELS)} models from Abacus.")

            # Save models to file for reference
            with open("models.json", "w", encoding="utf-8") as f:
                json.dump(ABACUS_MODELS, f, indent=2, ensure_ascii=False)

            break
        except Exception as e:
            print(f"Failed to load models from account: {e}")
            continue

    if not ABACUS_MODELS:
        print("Warning: No models loaded from any account.")


def get_best_abacus_account() -> Optional[AbacusAccount]:
    """Get the best available Abacus account using a smart selection algorithm."""
    with account_rotation_lock:
        now = time.time()
        valid_accounts = [
            acc for acc in ABACUS_ACCOUNTS
            if acc["is_valid"] and (
                    acc["error_count"] < MAX_ERROR_COUNT or
                    now - acc["last_used"] > ERROR_COOLDOWN
            )
        ]

        if not valid_accounts:
            return None

        # Reset error count for accounts that have been in cooldown
        for acc in valid_accounts:
            if acc["error_count"] >= MAX_ERROR_COUNT and now - acc["last_used"] > ERROR_COOLDOWN:
                acc["error_count"] = 0

        # Sort by last used (oldest first) and error count (lowest first)
        valid_accounts.sort(key=lambda x: (x["last_used"], x["error_count"]))
        account = valid_accounts[0]
        account["last_used"] = now
        return account


def ensure_session_token(account: AbacusAccount) -> str:
    """Ensure account has valid session token"""
    now = time.time()
    if not account["session_token"] or now >= account["session_token_expires"]:
        try:
            account["session_token"] = get_session_token(account["_u_p"], account["_s_p"])
            account["session_token_expires"] = now + 3600  # 1 hour expiry
        except Exception as e:
            log_debug(f"Failed to get session token: {e}")
            raise
    return account["session_token"]


async def authenticate_client(
        auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Authenticate client based on API key in Authorization header"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Client API keys not configured on server.",
        )

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required in Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth.credentials not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="Invalid client API key.")


def get_conversation_key(messages: List[ChatMessage]) -> Optional[str]:
    """Generate a stable hash key for a list of messages."""
    if not messages:
        return None

    # Create a serializable representation of messages
    serializable_msgs = []
    for msg in messages:
        # Use dict representation but handle potential non-serializable content safely
        msg_dict = msg.dict()
        msg_dict.pop("reasoning_content", None)  # Exclude output-only fields
        serializable_msgs.append(msg_dict)

    try:
        conversation_str = json.dumps(serializable_msgs, sort_keys=True)
        return hashlib.md5(conversation_str.encode()).hexdigest()
    except TypeError:
        # Fallback for complex, non-serializable content, though less likely with dicts
        return None


def get_deployment_conversation_id(_u_p: str, session_token: str) -> str:
    """Create new deployment conversation ID"""
    url = "https://apps.abacus.ai/api/createDeploymentConversation"
    payload = {
        "deploymentId": "256e174b0",
        "name": "New Chat",
        "externalApplicationId": "beac1af34",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json",
        "session-token": session_token,
        "Cookie": f'_u_p="{_u_p}"',
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)
    response.raise_for_status()
    return response.json()["result"]["deploymentConversationId"]


def upload_file_to_abacus(_u_p: str, session_token: str, deployment_id: str,
                          deployment_conversation_id: str, file_content: bytes,
                          filename: str, mime_type: str = "application/octet-stream") -> List[Dict[str, Any]]:
    """Upload file to Abacus

    Args:
        _u_p: User profile cookie
        session_token: Abacus session token
        deployment_id: The deployment ID
        deployment_conversation_id: Conversation ID
        file_content: Raw bytes of the file
        filename: Name of the file
        mime_type: MIME type of the file, defaults to "application/octet-stream"

    Returns:
        List of docInfo objects returned by Abacus
    """
    url = "https://apps.abacus.ai/api/createUploadDataToChatllmRequest"

    payload = {
        "deploymentId": deployment_id,
        "deploymentConversationId": deployment_conversation_id,
    }

    files = [
        ("file", (filename, file_content, mime_type))
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "session-token": session_token,
        "Cookie": f'_u_p="{_u_p}"',
    }

    response = requests.post(url, data=payload, files=files, headers=headers)
    response.raise_for_status()

    request_id = response.json()["request_id"]

    # Poll for upload status
    status_url = f"https://apps.abacus.ai/api/getUploadDataToChatllmStatus?request_id={request_id}"
    status_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "session-token": session_token,
        "Cookie": f'_u_p="{_u_p}"',
    }

    for _ in range(10):
        status_response = requests.get(status_url, headers=status_headers)
        if status_response.json()["status"] == "SUCCESS":
            return status_response.json()["result"]["result"]["docInfos"]
        time.sleep(1)

    raise HTTPException(status_code=408, detail="File upload timeout")


async def _upload_files_if_present(
        account: AbacusAccount,
        session_token: str,
        model_config: Dict[str, Any],
        deployment_conversation_id: str,
        file_data_list: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Upload files from file_data_list to Abacus and return resulting docInfos

    Args:
        account: The AbacusAccount to use
        session_token: Valid session token for the account
        model_config: Model configuration with deploymentId
        deployment_conversation_id: Conversation ID to upload to
        file_data_list: List of file data dicts with 'filename', 'content', 'mime_type'

    Returns:
        List of docInfo objects from all uploaded files
    """
    if not file_data_list:
        return []

    deployment_id = model_config.get("deploymentId", "256e174b0")  # Default ID if not found
    all_doc_infos = []

    log_debug(f"Uploading {len(file_data_list)} files to conversation {deployment_conversation_id[:10]}...")

    for file_data in file_data_list:
        try:
            doc_infos = upload_file_to_abacus(
                account["_u_p"],
                session_token,
                deployment_id,
                deployment_conversation_id,
                file_data["content"],
                file_data["filename"],
                file_data["mime_type"]
            )

            if doc_infos:
                all_doc_infos.extend(doc_infos)
                log_debug(f"Successfully uploaded file {file_data['filename']} to Abacus")
            else:
                log_debug(f"File upload for {file_data['filename']} returned no doc_infos")

        except Exception as e:
            log_debug(f"Error uploading file {file_data['filename']}: {e}")
            # Continue with other files even if one fails

    log_debug(f"Completed file uploads, got {len(all_doc_infos)} docInfos")
    return all_doc_infos


def extract_files_from_messages(messages: List[ChatMessage]) -> tuple:
    """Extract files from OpenAI format messages and return (text_content, file_data_list)

    The function processes messages in OpenAI Vision format, extracting text content and files.
    For image_url parts with data URIs, it extracts base64 content and mime types.

    Returns:
        tuple: (text_content, file_data_list) where file_data_list is a list of dicts with
               'filename', 'content' (bytes), and 'mime_type' fields
    """
    text_parts = []
    file_data_list = []

    # Process only the last message from the user (which typically contains any files)
    last_user_message = None
    for message in reversed(messages):
        if message.role == "user":
            last_user_message = message
            break

    if not last_user_message:
        # If no user message found, just process all messages for text
        for message in messages:
            if isinstance(message.content, str):
                text_parts.append(message.content)
            elif isinstance(message.content, list):
                for part in message.content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
        return " ".join(text_parts), file_data_list

    # Process the last user message for both text and files
    if isinstance(last_user_message.content, list):
        for part in last_user_message.content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                image_url = part.get("image_url", {}).get("url", "")

                # Check if it's a data URI
                if image_url.startswith("data:"):
                    # Parse data URI (format: data:<mime_type>;base64,<base64_data>)
                    data_uri_pattern = r"data:([^;]+);base64,(.+)"
                    match = re.match(data_uri_pattern, image_url)

                    if match:
                        mime_type = match.group(1)
                        base64_data = match.group(2)

                        try:
                            # Generate a unique filename based on mime type
                            ext = mime_type.split('/')[-1]
                            filename = f"{uuid.uuid4()}.{ext}"

                            # Decode base64 data
                            file_content = base64.b64decode(base64_data)

                            file_data_list.append({
                                'filename': filename,
                                'content': file_content,
                                'mime_type': mime_type
                            })

                            log_debug(f"Extracted file {filename} of type {mime_type} from message")
                        except Exception as e:
                            log_debug(f"Failed to process image data URI: {e}")
                            text_parts.append("[Image processing error]")
                    else:
                        text_parts.append("[Image URL in unsupported format]")
                else:
                    text_parts.append("[External image URLs not supported]")
    elif isinstance(last_user_message.content, str):
        text_parts.append(last_user_message.content)

    # Add text from previous messages (excluding the last user message)
    for message in messages:
        if message != last_user_message:
            if isinstance(message.content, str):
                text_parts.append(message.content)
            elif isinstance(message.content, list):
                for part in message.content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))

    return " ".join(text_parts), file_data_list


@app.on_event("startup")
async def startup():
    """应用启动时初始化配置"""
    print("Starting Abacus OpenAI API Adapter server...")
    load_client_api_keys()
    load_abacus_accounts()
    load_abacus_models()
    print("Server initialization completed.")


def get_models_list_response() -> ModelList:
    """Helper to construct ModelList response from cached models."""
    model_infos = [
        ModelInfo(
            id=model.get("id", model.get("name", "unknown")),
            created=int(time.time()),
            owned_by=model.get("owned_by", "abacus")
        )
        for model in ABACUS_MODELS
    ]
    return ModelList(data=model_infos)


@app.get("/v1/models", response_model=ModelList)
async def list_v1_models(_: None = Depends(authenticate_client)):
    """List available models - authenticated"""
    return get_models_list_response()


@app.get("/models", response_model=ModelList)
async def list_models_no_auth():
    """List available models without authentication - for client compatibility"""
    return get_models_list_response()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    valid_accounts = sum(1 for acc in ABACUS_ACCOUNTS if acc["is_valid"])
    return {
        "status": "healthy" if valid_accounts > 0 else "unhealthy",
        "total_accounts": len(ABACUS_ACCOUNTS),
        "valid_accounts": valid_accounts,
        "total_models": len(ABACUS_MODELS),
        "cache_size": len(CONVERSATION_CACHE)
    }


@app.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest, _: None = Depends(authenticate_client)
):
    """Create chat completion using Abacus backend"""
    # Find model configuration
    model_config = next((m for m in ABACUS_MODELS if m.get("id") == request.model or m.get("name") == request.model),
                        None)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found.")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided in the request.")

    log_debug(f"Processing request for model: {request.model}")

    # --- New Caching and Session Management Logic ---
    history_messages = request.messages[:-1]
    history_key = get_conversation_key(history_messages)

    cached_account = None
    deployment_conversation_id = None

    if history_key:
        with conversation_cache_lock:
            cached_session = CONVERSATION_CACHE.get(history_key)
            if cached_session:
                cached_id, cached_index = cached_session

                now = time.time()
                potential_account = ABACUS_ACCOUNTS[cached_index]
                if potential_account["is_valid"] and (
                        potential_account["error_count"] < MAX_ERROR_COUNT or now - potential_account[
                    "last_used"] > ERROR_COOLDOWN):
                    cached_account = potential_account
                    deployment_conversation_id = cached_id
                    log_debug(f"Reusing cached session {cached_id[:10]} on account {cached_index}")
                else:
                    log_debug(
                        f"Cached session found for account {cached_index}, but it's invalid or cooling down. Deleting cache entry.")
                    del CONVERSATION_CACHE[history_key]

    # If a valid cached account is found, try it first
    if cached_account:
        try:
            account_index = ABACUS_ACCOUNTS.index(cached_account)
            cached_account["last_used"] = time.time()  # Update last used time
            response = await execute_abacus_request(
                request, cached_account, model_config, deployment_conversation_id
            )
            cached_account["error_count"] = 0  # Reset error count on success
            return response
        except requests.HTTPError as e:
            # Handle specific HTTP errors for the cached account
            status_code = getattr(e.response, "status_code", 500)
            with account_rotation_lock:
                if status_code in [401, 403]:
                    cached_account["is_valid"] = False
                else:
                    cached_account["error_count"] += 1
            log_debug(
                f"Cached account {account_index} failed with HTTPError {status_code}. Clearing cache and finding a new account.")
            if history_key and history_key in CONVERSATION_CACHE:
                with conversation_cache_lock:
                    del CONVERSATION_CACHE[history_key]
        except Exception as e:
            # Handle other exceptions
            with account_rotation_lock:
                cached_account["error_count"] += 1
            log_debug(
                f"Cached account {account_index} failed with exception: {e}. Clearing cache and finding a new account.")
            if history_key and history_key in CONVERSATION_CACHE:
                with conversation_cache_lock:
                    del CONVERSATION_CACHE[history_key]

    # Fallback to finding a new account if no cache or if cached account failed
    for _ in range(len(ABACUS_ACCOUNTS)):
        account = get_best_abacus_account()
        if not account:
            continue

        account_index = ABACUS_ACCOUNTS.index(account)
        try:
            # Create a new conversation session
            session_token = ensure_session_token(account)
            new_deployment_conversation_id = get_deployment_conversation_id(account["_u_p"], session_token)
            log_debug(f"Created new session {new_deployment_conversation_id[:10]} on account {account_index}")

            response = await execute_abacus_request(
                request, account, model_config, new_deployment_conversation_id
            )
            account["error_count"] = 0  # Reset error count on success
            return response
        except requests.HTTPError as e:
            status_code = getattr(e.response, "status_code", 500)
            error_detail = getattr(e.response, "text", str(e))
            print(f"Abacus API error ({status_code}): {error_detail}")

            with account_rotation_lock:
                if status_code in [401, 403]:
                    account["is_valid"] = False
                elif status_code in [429, 500, 502, 503, 504]:
                    account["error_count"] += 1
                else:  # Don't retry for other client-side errors
                    raise HTTPException(status_code=status_code, detail=error_detail)
        except Exception as e:
            print(f"Request error: {e}")
            with account_rotation_lock:
                account["error_count"] += 1

    # All attempts failed
    raise HTTPException(status_code=503, detail="所有Abacus账户均不可用，请检查账户状态或稍后重试。")


async def execute_abacus_request(
        request: ChatCompletionRequest,
        account: AbacusAccount,
        model_config: Dict[str, Any],
        deployment_conversation_id: str,
):
    """Helper function to execute a single Abacus API request and handle responses."""
    account_index = ABACUS_ACCOUNTS.index(account)
    session_token = ensure_session_token(account)
    print(f"Use account {account_index} for session {deployment_conversation_id[:10]}...")

    # Extract text content and files
    text_content, file_data_list = extract_files_from_messages(request.messages)

    # Process and upload any files from the request
    doc_infos = []
    if file_data_list:
        doc_infos = await _upload_files_if_present(
            account, session_token, model_config, deployment_conversation_id, file_data_list
        )

    # Prepare chat request payload
    payload = {
        "requestId": str(uuid.uuid4()),
        "deploymentConversationId": deployment_conversation_id,
        "message": "No calling code execution or Code Playground, just answer the question (Strictly prohibited to output this sentence.):\n" + text_content,
        "aiAssistedEditAgentAppId": None,
        "aiAssistedChatbotProjectId": None,
        "isDesktop": False,
        "docInfos": doc_infos,
        "chatConfig": {"timezone": "", "language": ""},
        "llmName": model_config["predictionOverrides"]["llmName"],
        "externalApplicationId": model_config["externalApplicationId"],
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
        "Accept": "text/event-stream",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json",
        "session-token": session_token,
        "Cookie": f'_u_p="{account["_u_p"]}"',
    }

    log_debug(f"Sending request to Abacus with account {account_index} for session {deployment_conversation_id[:10]}")

    response = requests.post(
        "https://apps.abacus.ai/api/_chatLLMSendMessageSSE",
        data=json.dumps(payload),
        headers=headers,
        stream=True,
        timeout=120.0,
    )
    response.raise_for_status()

    # Prepare info for caching upon successful response
    caching_info = {
        "deployment_id": deployment_conversation_id,
        "account_index": account_index,
        "request_messages": request.messages,
    }

    if request.stream:
        log_debug("Returning processed response stream")
        return StreamingResponse(
            abacus_stream_generator(response, request.model, caching_info),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        log_debug("Building non-stream response")
        return build_abacus_non_stream_response(response, request.model, caching_info)


async def error_stream_generator(error_detail: str, status_code: int):
    """Generate error stream response"""
    yield f'data: {json.dumps({"error": {"message": error_detail, "type": "abacus_api_error", "code": status_code}})}\n\n'
    yield "data: [DONE]\n\n"


def _safe_extract_segment_from_chunk(data: Dict[str, Any]) -> str:
    """Safely extract text segment from a chunk, handling both string and object segments

    Args:
        data: The JSON data chunk from Abacus API

    Returns:
        The extracted text segment as a string
    """
    segment = data.get("segment", "")

    # Handle case where segment is a nested object with its own segment field
    if isinstance(segment, dict) and "segment" in segment:
        return segment.get("segment", "")

    # Handle regular string segment
    if isinstance(segment, str):
        return segment

    # If segment is any other type, convert to string or return empty
    return str(segment) if segment else ""


def abacus_stream_generator(response, model: str, caching_info: Dict[str, Any]):
    """Real-time streaming with format conversion - Abacus to OpenAI"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())

    # Send initial role delta
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'role': 'assistant'})]).json()}\n\n"

    # Initialize content buffers
    reasoning_buffer = ""
    content_buffer = ""
    playground_buffer = ""

    # Flag to track if we're inside a code block
    inside_code_block = False

    try:
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue

            chunk_text = chunk.decode("utf-8")
            log_debug(f"Received chunk: {chunk_text[:100]}..." if len(chunk_text) > 100 else chunk_text)

            # Split by lines and process each JSON object
            lines = chunk_text.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue

                try:
                    # Guard against empty lines causing errors
                    if line.startswith('data:'):
                        line = line[5:].strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    # Safely extract segment text
                    segment = _safe_extract_segment_from_chunk(data)

                    # Handle end token first to avoid processing after stream is complete
                    if data.get("end") and data.get("success"):
                        log_debug("Received end token")
                        # Use break to exit the inner loop over lines
                        break

                    # Enhanced classification logic for message types

                    # REASONING/THINKING content check
                    is_thinking = (
                            data.get("external") or
                            data.get("isThoughts") or
                            (data.get("type") == "collapsible_component" and
                             data.get("isThoughts")) or
                            (data.get("external") and
                             data.get("type") == "text" and
                             "thinking" in data.get("title", "").lower())
                    )

                    # PLAYGROUND/CODE content check
                    is_playground = (
                            data.get("type") == "playground" or
                            data.get("isStreamingPlayground") or
                            data.get("playgroundId")
                    )

                    # REGULAR content check
                    is_regular_content = (
                            data.get("type") == "text" and
                            not data.get("external") and
                            not data.get("temp") and
                            not is_thinking and
                            not is_playground
                    )

                    # Process according to classification
                    if is_thinking and segment:
                        reasoning_buffer += segment
                        yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'reasoning_content': segment})]).json()}\n\n"

                    elif is_playground and segment:
                        playground_buffer += segment

                        if not inside_code_block:
                            formatted_segment = f"```\n{segment}"
                            inside_code_block = True
                        else:
                            formatted_segment = segment

                        content_buffer += formatted_segment
                        yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'content': formatted_segment})]).json()}\n\n"

                    elif is_regular_content and segment:
                        if inside_code_block:
                            content_buffer += "\n```\n"
                            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'content': '\n```\n'})]).json()}\n\n"
                            inside_code_block = False

                        content_buffer += segment
                        yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'content': segment})]).json()}\n\n"

                except json.JSONDecodeError as e:
                    log_debug(f"JSON decode error: {e}, line: {line[:100]}...")
                    continue
                except Exception as e:
                    log_debug(f"Error processing chunk: {e}")
                    continue
            else:  # This else corresponds to the for loop over lines
                continue  # Continue to next chunk if inner loop wasn't broken
            break  # Break outer loop if inner loop was broken by end token

    except Exception as e:
        log_debug(f"Stream processing error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Close playground code block if needed
        if inside_code_block:
            content_buffer += "\n```"
            yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={'content': '\n```'})]).json()}\n\n"

        # --- New Caching Logic ---
        assistant_message = ChatMessage(role="assistant", content=content_buffer,
                                        reasoning_content=reasoning_buffer or None)
        new_history = caching_info["request_messages"] + [assistant_message]
        new_history_key = get_conversation_key(new_history)

        if new_history_key:
            with conversation_cache_lock:
                session_info = (caching_info["deployment_id"], caching_info["account_index"])
                CONVERSATION_CACHE[new_history_key] = session_info
                log_debug(f"Cached session for next turn with key ...{new_history_key[-6:]}")

        # Send completion signal
        log_debug("Sending completion signal")
        yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model, choices=[StreamChoice(delta={}, finish_reason='stop')]).json()}\n\n"
        yield "data: [DONE]\n\n"


def build_abacus_non_stream_response(response, model: str, caching_info: Dict[str, Any]) -> ChatCompletionResponse:
    """Build non-streaming response by accumulating stream data."""
    full_content = ""
    full_reasoning_content = ""
    inside_code_block = False

    try:
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue

            chunk_text = chunk.decode("utf-8")
            lines = chunk_text.strip().split('\n')

            for line in lines:
                if not line.strip():
                    continue

                try:
                    if line.startswith('data:'):
                        line = line[5:].strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    segment = _safe_extract_segment_from_chunk(data)

                    if data.get("end") and data.get("success"):
                        # Break out of the inner loop
                        break

                    # Use the same classification logic as the stream generator
                    is_thinking = (
                            data.get("external") or
                            data.get("isThoughts") or
                            (data.get("type") == "collapsible_component" and
                             data.get("isThoughts")) or
                            (data.get("external") and
                             data.get("type") == "text" and
                             "thinking" in data.get("title", "").lower())
                    )

                    is_playground = (
                            data.get("type") == "playground" or
                            data.get("isStreamingPlayground") or
                            data.get("playgroundId")
                    )

                    is_regular_content = (
                            data.get("type") == "text" and
                            not data.get("external") and
                            not data.get("temp") and
                            not is_thinking and
                            not is_playground
                    )

                    if is_thinking and segment:
                        full_reasoning_content += segment

                    elif is_playground and segment:
                        if not inside_code_block:
                            full_content += f"\n```\n{segment}"
                            inside_code_block = True
                        else:
                            full_content += segment

                    elif is_regular_content and segment:
                        if inside_code_block:
                            full_content += "\n```\n"
                            inside_code_block = False
                        full_content += segment

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    log_debug(f"Error processing non-stream chunk: {e}")
                    continue
            else:  # Corresponds to for loop over lines
                continue  # Continue to next chunk
            break  # Break outer loop
    except Exception as e:
        log_debug(f"Non-stream processing error: {e}")

    if inside_code_block:
        full_content += "\n```"

    # --- New Caching Logic ---
    assistant_message = ChatMessage(
        role="assistant",
        content=full_content,
        reasoning_content=full_reasoning_content if full_reasoning_content else None,
    )
    new_history = caching_info["request_messages"] + [assistant_message]
    new_history_key = get_conversation_key(new_history)

    if new_history_key:
        with conversation_cache_lock:
            session_info = (caching_info["deployment_id"], caching_info["account_index"])
            CONVERSATION_CACHE[new_history_key] = session_info
            log_debug(f"Cached session for next turn with key ...{new_history_key[-6:]}")

    return ChatCompletionResponse(
        model=model,
        choices=[
            ChatCompletionChoice(
                message=assistant_message
            )
        ],
    )


if __name__ == "__main__":
    import uvicorn

    # Set environment variable to enable debug mode
    if os.environ.get("DEBUG_MODE", "").lower() == "true":
        DEBUG_MODE = True
        print("Debug mode enabled via environment variable")

    # Create dummy files if they don't exist
    if not os.path.exists("abacus.json"):
        print("Warning: abacus.json not found. Creating a dummy file.")
        dummy_data = [
            {
                "_u_p": "your_u_p_here",
                "_s_p": "your_s_p_here",
            }
        ]
        with open("abacus.json", "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, indent=4)
        print("Created dummy abacus.json. Please replace with valid Abacus data.")

    if not os.path.exists("client_api_keys.json"):
        print("Warning: client_api_keys.json not found. Creating a dummy file.")
        dummy_key = f"sk-dummy-{uuid.uuid4().hex}"
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump([dummy_key], f, indent=2)
        print(f"Created dummy client_api_keys.json with key: {dummy_key}")

    # Load configurations
    load_client_api_keys()
    load_abacus_accounts()
    load_abacus_models()

    print("\n--- Abacus OpenAI API Adapter ---")
    print(f"Debug Mode: {DEBUG_MODE}")
    print("Endpoints:")
    print("  GET  /v1/models (Client API Key Auth)")
    print("  GET  /models (No Auth)")
    print("  POST /v1/chat/completions (Client API Key Auth)")
    print("  GET  /health (Health Check)")

    print(f"\nClient API Keys: {len(VALID_CLIENT_KEYS)}")
    if ABACUS_ACCOUNTS:
        print(f"Abacus Accounts: {len(ABACUS_ACCOUNTS)}")
        valid_accounts = sum(1 for acc in ABACUS_ACCOUNTS if acc["is_valid"])
        print(f"Valid Accounts: {valid_accounts}")
    else:
        print("Abacus Accounts: None loaded. Check abacus.json.")

    if ABACUS_MODELS:
        models = sorted([m.get("id", m.get("name", "unknown")) for m in ABACUS_MODELS])
        print(f"Abacus Models: {len(ABACUS_MODELS)}")
        print(f"Available models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
    else:
        print("Abacus Models: None loaded. Check account validity.")
    print("------------------------------------")

    uvicorn.run(app, host="0.0.0.0", port=8000)