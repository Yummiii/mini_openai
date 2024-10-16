use mini_openai;

#[derive(Debug, serde::Deserialize)]
struct Hello {
    hello: String,
}

fn main() -> Result<(), mini_openai::Error> {
    let client = mini_openai::Client::new(None, None)?;

    // Create a new chat completion request
    let mut request = mini_openai::ChatCompletions::default();

    // Add a message to the chat history
    request.messages.push(mini_openai::Message {
        content: r#"Respond in JSON: {"hello": "a word of your choosing"}"#.to_string(),
        role: mini_openai::ROLE_USER.to_string(),
    });

    // Send the request to the OpenAI API
    let response: Hello =
        client.chat_completions_into(&request, 3, mini_openai::parse_json_lenient)?;

    // Print the generated completion
    println!("{:?}", response);

    Ok(())
}
