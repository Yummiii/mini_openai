# mini_openai

A Minimal-Dependency Rust Crate for OpenAI-compatible LLM Servers

## Overview

`mini_openai` is a lightweight Rust crate that provides a simple and minimal-dependency way to interact with OpenAI-compatible LLM (Large Language Model) servers. With this crate, you can easily integrate OpenAI's language models into your Rust applications without adding unnecessary dependencies or complexity.

## Features

* **Simple and minimalistic API**: A straightforward and easy-to-use API for sending requests to OpenAI-compatible LLM servers.
* **Synchronous or Asynchronous**: This crate supports both `reqwest` and `ureq` HTTP clients, allowing you to choose the one that best fits your needs.
* **Custom Endpoints First**: mini_openai supports configuration via environment variables, making it easy to switch between different API keys and base URLs. This includes support for the official OpenAI services, but also locally hosted ones via text-generation-webui, ollama, vLLM, etc.

## Getting Started

To get started with mini_openai, add the following dependency to your `Cargo.toml` file:

```toml
[dependencies]
mini_openai = "*"
```

Then, import the crate in your Rust code and create a new client instance:

```rust
use mini_openai;

fn main() -> Result<(), mini_openai::Error> {
    let client = mini_openai::Client::new(None, None)?;

    // Create a new chat completion request
    let mut request = mini_openai::ChatCompletions::default();

    // Add a message to the chat history
    request.messages.push(mini_openai::Message {
        content: "Who are you?".to_string(),
        role: mini_openai::ROLE_USER.to_string(),
    });

    // Send the request to the OpenAI API
    let response = client.chat_completions(&request)?;

    // Print the generated completion
    println!("{}", response.choices[0].message.content);

    Ok(())
}
```

You can also run the examples:

```sh
# Modify as you require:
export OPENAI_API_BASE=http://localhost:3000/v1
# export OPENAI_API_KEY=sk-...

# Synchronous API with ureq:
cargo run --features=ureq --example sync-chat-completion

# Asynchronous API with reqwest:
cargo run --features=reqwest --example async-chat-completion
```

### Configuration

Your users can set the following environment variables for configuration. You, the developer, can then use either `Client::new()` or `Client::new_from_environment()` to abide to these variables. If set, they'll take precedence over your parameters passed to `Client::new()`!

* `OPENAI_API_KEY`: The API-Key. Optional if not using OpenAI services.
* `OPENAI_API_BASE`: Base URL to the API (Example: `http://localhost:3000/v1` - Note the `/v1`!)

If you don't intend for your user to take control, you can use `Client::new_without_environment()` instead.

## Contributing

mini_openai is an open-source project, and we welcome contributions from the community. If you'd like to report a bug, suggest a new feature, or help with development, please open an issue or submit a pull request on our GitHub page.

## License

mini_openai is licensed under the BSD 2-clause License. See the `LICENSE` file for details.

## Acknowledgments

A Thank You goes to the maintainers of `reqwest` and `ureq` for their excellent HTTP client libraries. I wrote this library because I wanted a small and easy-to-use library to interact with LLMs without pulling in dependencies for features I don't need.
