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
