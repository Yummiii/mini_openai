use std::env;

#[cfg(all(feature = "reqwest", feature = "ureq"))]
compile_error!("Features 'reqwest' and 'ureq' are mutually exclusive.");

#[cfg(not(any(feature = "reqwest", feature = "ureq")))]
compile_error!("One of the features 'reqwest' and 'ureq' must be enabled.");

use serde::ser::{SerializeMap, SerializeSeq};
#[cfg(feature = "ureq")]
use ureq;

#[cfg(feature = "reqwest")]
use reqwest;

const OPENAI_API_KEY: &str = "OPENAI_API_KEY";
const OPENAI_API_BASE: &str = "OPENAI_API_BASE";
const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("The configuration contains errors: {0}")]
    BadConfigurationError(String),

    #[error("Failed to serialize response: {0}")]
    SerializationError(serde_json::Error),

    #[error("Failed to deserialize response: {0}")]
    DeserializationError(serde_json::Error),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("API error: {0}")]
    ApiError(String),
}

pub const DEFAULT_CHAT_MODEL: &str = "gpt-4o-mini";
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-ada-002";

pub const ROLE_SYSTEM: &str = "system";
pub const ROLE_USER: &str = "user";
pub const ROLE_ASSISTANT: &str = "assistant";

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub content: String,
    pub role: String,
}

#[derive(Debug)]
pub enum ResponseFormat {
    JsonObject,
    JsonSchema(serde_json::Value),
}

impl serde::Serialize for ResponseFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ResponseFormat::JsonObject => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", "json_object")?;
                map.end()
            }
            ResponseFormat::JsonSchema(schema) => {
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry("type", "json_schema")?;
                map.serialize_entry("json_schema", schema)?;
                map.end()
            }
        }
    }
}

#[derive(Debug)]
pub enum Stop {
    String(String),
    Array(Vec<String>),
}

impl serde::Serialize for Stop {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Stop::String(string) => serializer.serialize_str(&string),
            Stop::Array(strings) => {
                let mut array = serializer.serialize_seq(Some(strings.len()))?;

                for string in strings {
                    array.serialize_element(string)?;
                }

                array.end()
            }
        }
    }
}

fn is_false(value: &bool) -> bool {
    *value == false
}

// NOTE: As we're supporting non-OpenAI API implementations, we should only
// send options in requests that are set. Some implementations don't like if
// they see options they don't know, even if they're "null".

/// Chat completions structure.
///
/// For reference, see: https://platform.openai.com/docs/api-reference/chat
///
/// To construct this structure easily use the default trait:
///
/// ```rust
/// let request = mini_openai::ChatCompletions {
///   messages: vec![
///     mini_openai::Message{
///         role: mini_openai::ROLE_SYSTEM.into(),
///         content: "Who are you?".into()
///     }
///   ],
///   ..Default::default()
/// };
/// ```
#[derive(Debug, serde::Serialize)]
pub struct ChatCompletions {
    pub messages: Vec<Message>,
    pub model: String,
    #[serde(skip_serializing_if = "is_false")]
    pub store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "is_false")]
    pub logprobs: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Stop>,
    /// Must be 'false': Only non-streaming is supported.
    pub stream: bool,
    // pub stream_options: Option<>,
    // pub tools: Option<>,
    // pub tool_choice: Option<>,
    // pub parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl Default for ChatCompletions {
    fn default() -> Self {
        Self {
            messages: Default::default(),
            model: DEFAULT_CHAT_MODEL.into(),
            store: false,
            metadata: None,
            logit_bias: None,
            logprobs: false,
            top_logprobs: None,
            max_tokens: None,
            max_completion_tokens: None,
            n: None,
            presence_penalty: None,
            response_format: None,
            seed: None,
            service_tier: None,
            stop: None,
            stream: false,
            user: None,
        }
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    //pub logprobs: Option<Logprobs>,
    pub finish_reason: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct ChatCompletionsResponse {
    pub id: String,
    pub object: String,
    pub created: usize,
    pub model: String,
    pub choices: Vec<Choice>,
    //pub usage: Usage,
}

#[derive(Debug)]
pub enum Input {
    String(String),
    Array(Vec<String>),
}

impl From<String> for Input {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for Input {
    fn from(value: &str) -> Self {
        Self::String(value.to_string())
    }
}

impl From<Vec<String>> for Input {
    fn from(values: Vec<String>) -> Self {
        Self::Array(values)
    }
}

impl From<&[String]> for Input {
    fn from(values: &[String]) -> Self {
        Self::Array(values.to_vec())
    }
}

impl serde::Serialize for Input {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Input::String(string) => serializer.serialize_str(string),
            Input::Array(array) => {
                let mut seq = serializer.serialize_seq(Some(array.len()))?;
                for s in array {
                    seq.serialize_element(s)?;
                }
                seq.end()
            }
        }
    }
}

/// Embeddings request structure.
///
/// You can easily construct the input using .into():
///
/// ```rust
/// let embeddings = mini_openai::Embeddings {
///     input: "Hello".into(),
///     ..Default::default()
/// };
/// ```
///
#[derive(Debug, serde::Serialize)]
pub struct Embeddings {
    pub input: Input,
    pub model: String,
    //pub encoding_format: EncodingFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

impl Default for Embeddings {
    fn default() -> Self {
        Self {
            input: Input::String("".into()),
            model: DEFAULT_EMBEDDING_MODEL.into(),
            dimensions: None,
            user: None,
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct EmbeddingsResponse {
    data: Vec<Embedding>,
    model: String,
    usage: Option<Usage>, // Not all implementations may return this
}

#[derive(Debug, serde::Deserialize)]
struct Embedding {
    index: u64,
    embedding: Vec<f64>,
}

#[derive(Debug, serde::Deserialize)]
struct Usage {
    prompt_tokens: u32,
    total_tokens: u32,
}

#[cfg(feature = "ureq")]
struct ClientImpl {
    client: ureq::Agent,
    token: Option<String>,
}

#[cfg(feature = "ureq")]
impl ClientImpl {
    fn new(token: Option<String>) -> Result<ClientImpl, Error> {
        Ok(Self {
            client: ureq::Agent::new(),
            token,
        })
    }

    fn do_request(&self, url: String, body: String) -> Result<String, Error> {
        let mut request = self
            .client
            .post(&url)
            .set("Content-Type", "application/json");

        if let Some(token) = self.token.as_ref() {
            request = request.set("Authorization", &format!("Bearer {}", token));
        }

        let response = request
            .send_string(&body)
            .map_err(|e| Error::NetworkError(e.to_string()))?;

        if response.status() != 200 {
            let text = format!("{} {}", response.status(), response.status_text());
            Err(Error::ApiError(text))?;
        }

        let body = response
            .into_string()
            .map_err(|e| Error::NetworkError(e.to_string()))?;
        Ok(body)
    }
}

#[cfg(feature = "reqwest")]
struct ClientImpl {
    client: reqwest::Client,
}

#[cfg(feature = "reqwest")]
impl ClientImpl {
    fn new(token: Option<String>) -> Result<ClientImpl, Error> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(token) = token {
            let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token))
                .map_err(|e| Error::BadConfigurationError(e.to_string()))?;
            value.set_sensitive(true);
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }

        let client = reqwest::ClientBuilder::new()
            .default_headers(headers)
            .build()
            .map_err(|e| Error::BadConfigurationError(e.to_string()))?;

        Ok(Self { client })
    }

    async fn do_request(&self, url: String, body: String) -> Result<String, Error> {
        let response = self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| Error::NetworkError(e.to_string()))?
            .error_for_status()
            .map_err(|e| Error::ApiError(e.to_string()))?
            .text()
            .await
            .map_err(|e| Error::NetworkError(e.to_string()))?;

        Ok(response)
    }
}

pub struct Client {
    inner: ClientImpl,
    base_uri: String,
}

impl Client {
    /// Creates a new `Client` instance.
    ///
    /// This function will first check for environment variables `OPENAI_API_BASE` and `OPENAI_API_KEY`.
    /// If they are not set, it will use the provided `base_uri` and `token` parameters. If neither are set,
    /// it will use the default API base URI.
    ///
    /// If a `token` is not provided and `base_uri` is set to the OpenAI API base URI, an error will be returned.
    ///
    /// # Arguments
    ///
    /// * `base_uri`: The base URI of the API, or `None` to use the environment variable or default.
    /// * `token`: The API token, or `None` to use the environment variable.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `Client` instance, or an `Error` if the configuration is invalid.
    pub fn new(base_uri: Option<String>, token: Option<String>) -> Result<Client, Error> {
        let env_base_uri = env::var(OPENAI_API_BASE).unwrap_or_default();
        let env_token = env::var(OPENAI_API_KEY).unwrap_or_default();

        let base_uri = if env_base_uri.is_empty() {
            if let Some(uri) = base_uri {
                uri
            } else {
                DEFAULT_API_BASE.to_string()
            }
        } else {
            env_base_uri
        };

        let token = if env_token.is_empty() {
            token
        } else {
            Some(env_token)
        };

        Self::new_without_environment(base_uri, token)
    }

    /// Creates a new `Client` instance without checking environment variables.
    ///
    /// This function is used internally by `new` to create a client without checking for environment variables.
    ///
    /// # Arguments
    ///
    /// * `base_uri`: The base URI of the API.
    /// * `token`: The API token, or `None` if not required.
    ///
    /// # Returns
    ///
    /// If `base_uri` is empty, an error will be returned.
    /// If `base_uri` is set to the OpenAI API base URI and `token` is `None`, an error will be returned.
    /// A `Result` containing the new `Client` instance, or an `Error` if the configuration is invalid.
    pub fn new_without_environment(
        base_uri: String,
        token: Option<String>,
    ) -> Result<Client, Error> {
        if base_uri.is_empty() {
            return Err(Error::BadConfigurationError("No base URI given".into()));
        }

        // Only check if there's a token if we're connecting to OpenAI.
        // Custom endpoints may not require it, so don't enforce it for them.
        if base_uri == DEFAULT_API_BASE && token.is_none() {
            return Err(Error::BadConfigurationError("Missing api token".into()));
        }

        let inner = ClientImpl::new(token)?;
        Ok(Self { inner, base_uri })
    }

    /// Creates a new `Client` instance from environment variables.
    ///
    /// This function will read the `OPENAI_API_BASE` and `OPENAI_API_KEY` environment variables and use them to create a client.
    ///
    /// # Returns
    ///
    /// A `Result` containing the new `Client` instance, or an `Error` if the environment variables are not set.
    pub fn new_from_environment() -> Result<Client, Error> {
        let env_base_uri =
            env::var(OPENAI_API_BASE).map_err(|e| Error::BadConfigurationError(e.to_string()))?;
        let env_token = env::var(OPENAI_API_KEY).unwrap_or_default();

        let token = if env_token.is_empty() {
            None
        } else {
            Some(env_token)
        };

        Self::new_without_environment(env_base_uri, token)
    }

    /// Sends a request to the OpenAI API to generate a completion for a chat conversation.
    ///
    /// This function takes a `ChatCompletions` struct as input, which defines the parameters of the completion request,
    /// including the chat history, model to use, and desired response format.
    ///
    /// The function returns a `ChatCompletionsResponse` struct, which contains the generated completion.
    ///
    /// # Arguments
    ///
    /// * `request`: The `ChatCompletions` struct containing the request parameters.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `ChatCompletionsResponse` struct, or an `Error` if the request fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use mini_openai::{Client, ChatCompletions, Message, ROLE_USER};
    ///
    /// let client = Client::new(None, None).unwrap();
    ///
    /// // Create a new chat completion request
    /// let mut request = ChatCompletions::default();
    ///
    /// // Add a message to the chat history
    /// request.messages.push(Message {
    ///     content: "Hello!".to_string(),
    ///     role: ROLE_USER.to_string(),
    /// });
    ///
    /// // Send the request to the OpenAI API
    /// let response = client.chat_completions(&request).await.unwrap();
    ///
    /// // Print the generated completion
    /// println!("{}", response.choices[0].message.content);
    /// ```
    #[cfg(feature = "reqwest")]
    pub async fn chat_completions(
        &self,
        request: &ChatCompletions,
    ) -> Result<ChatCompletionsResponse, Error> {
        let url = format!("{}/chat/completions", self.base_uri);
        let body = serde_json::to_string(request).map_err(Error::SerializationError)?;
        let response = self.inner.do_request(url, body).await?;

        serde_json::from_str(&response).map_err(Error::DeserializationError)
    }

    /// Sends a request to the OpenAI API to generate a completion for a chat conversation.
    ///
    /// This function takes a `ChatCompletions` struct as input, which defines the parameters of the completion request,
    /// including the chat history, model to use, and desired response format.
    ///
    /// The function returns a `ChatCompletionsResponse` struct, which contains the generated completion.
    ///
    /// # Arguments
    ///
    /// * `request`: The `ChatCompletions` struct containing the request parameters.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `ChatCompletionsResponse` struct, or an `Error` if the request fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mini_openai::{Client, ChatCompletions, Message, ROLE_USER};
    ///
    /// let client = Client::new(None, None).unwrap();
    ///
    /// // Create a new chat completion request
    /// let mut request = ChatCompletions::default();
    ///
    /// // Add a message to the chat history
    /// request.messages.push(Message {
    ///     content: "Hello!".to_string(),
    ///     role: ROLE_USER.to_string(),
    /// });
    ///
    /// // Send the request to the OpenAI API
    /// let response = client.chat_completions(&request).unwrap();
    ///
    /// // Print the generated completion
    /// println!("{}", response.choices[0].message.content);
    /// ```
    #[cfg(feature = "ureq")]
    pub fn chat_completions(
        &self,
        request: &ChatCompletions,
    ) -> Result<ChatCompletionsResponse, Error> {
        let url = format!("{}/chat/completions", self.base_uri);
        let body = serde_json::to_string(request).map_err(Error::SerializationError)?;
        let response = self.inner.do_request(url, body)?;

        serde_json::from_str(&response).map_err(Error::DeserializationError)
    }

    //async fn completions(&self, request: &ChatCompletions) -> Result<CompletionsResponse, Error> {}

    /// Sends a request to the OpenAI API to generate embeddings of text.
    ///
    /// This function takes a `Embeddings` struct as input..
    ///
    /// The function returns a `EmbeddingsResponse` struct, which contains the generated embeddings.
    ///
    /// # Arguments
    ///
    /// * `request`: The `Embeddings` struct containing the request parameters.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `EmbeddingsResponse` struct, or an `Error` if the request fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use mini_openai::{Client, Embeddings, Message, ROLE_USER};
    ///
    /// let client = Client::new(None, None).unwrap();
    ///
    /// // Create a new chat completion request
    /// let request = Embeddings { input: "Hello".into(), ..Default::default() };
    ///
    /// // Send the request to the OpenAI API
    /// let response = client.embeddings(&request).await.unwrap();
    ///
    /// // Print the generated completion
    /// println!("{}", response.data[0].embedding);
    /// ```
    #[cfg(feature = "reqwest")]
    async fn embeddings(&self, request: &Embeddings) -> Result<EmbeddingsResponse, Error> {
        let url = format!("{}/embeddings", self.base_uri);
        let body = serde_json::to_string(request).map_err(Error::SerializationError)?;
        let response = self.inner.do_request(url, body).await?;

        serde_json::from_str(&response).map_err(Error::DeserializationError)
    }

    /// Sends a request to the OpenAI API to generate embeddings of text.
    ///
    /// This function takes a `Embeddings` struct as input..
    ///
    /// The function returns a `EmbeddingsResponse` struct, which contains the generated embeddings.
    ///
    /// # Arguments
    ///
    /// * `request`: The `Embeddings` struct containing the request parameters.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `EmbeddingsResponse` struct, or an `Error` if the request fails.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use mini_openai::{Client, Embeddings, Message, ROLE_USER};
    ///
    /// let client = Client::new(None, None).unwrap();
    ///
    /// // Create a new chat completion request
    /// let request = Embeddings { input: "Hello".into(), ..Default::default() };
    ///
    /// // Send the request to the OpenAI API
    /// let response = client.embeddings(&request).unwrap();
    ///
    /// // Print the generated completion
    /// println!("{}", response.data[0].embedding);
    /// ```
    #[cfg(feature = "ureq")]
    fn embeddings(&self, request: &Embeddings) -> Result<EmbeddingsResponse, Error> {
        let url = format!("{}/embeddings", self.base_uri);
        let body = serde_json::to_string(request).map_err(Error::SerializationError)?;
        let response = self.inner.do_request(url, body)?;

        serde_json::from_str(&response).map_err(Error::DeserializationError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ureq")]
    #[test]
    fn chat_completions() -> Result<(), Error> {
        let client = Client::new(None, None)?;
        let request = ChatCompletions {
            messages: vec![Message {
                role: ROLE_SYSTEM.into(),
                content: "Just say OK.".into(),
            }],
            ..Default::default()
        };

        let response: ChatCompletionsResponse = client.chat_completions(&request)?;

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content.contains("OK"), true);

        Ok(())
    }

    #[cfg(feature = "ureq")]
    #[test]
    fn embeddings() -> Result<(), Error> {
        let client = Client::new(None, None)?;
        let request = Embeddings {
            input: "Hello".into(),
            ..Default::default()
        };

        let response: EmbeddingsResponse = client.embeddings(&request)?;

        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].embedding.is_empty(), false);

        Ok(())
    }

    #[cfg(feature = "reqwest")]
    #[tokio::test]
    async fn chat_completions() -> Result<(), Error> {
        let client = Client::new(None, None)?;
        let request = ChatCompletions {
            messages: vec![Message {
                role: ROLE_SYSTEM.into(),
                content: "Just say OK.".into(),
            }],
            ..Default::default()
        };

        let response: ChatCompletionsResponse = client.chat_completions(&request).await?;

        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content.contains("OK"), true);

        Ok(())
    }

    #[cfg(feature = "reqwest")]
    #[test]
    fn embeddings() -> Result<(), Error> {
        let client = Client::new(None, None)?;
        let request = Embeddings {
            input: "Hello".into(),
            ..Default::default()
        };

        let response: EmbeddingsResponse = client.embeddings(&request).await?;

        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].embedding.is_empty(), false);

        Ok(())
    }
}
