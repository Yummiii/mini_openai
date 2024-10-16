use mini_openai;

#[tokio::main]
async fn main() -> Result<(), mini_openai::Error> {
    let client = mini_openai::Client::new(None, None)?;

    let request = mini_openai::Embeddings {
        input: "Hello".into(),
        ..Default::default()
    };

    let response = client.embeddings(&request).await?;
    println!("{:?}", response);

    Ok(())
}
