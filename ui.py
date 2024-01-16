import gradio as gr

from chain import Chain

llm_chain = Chain()

# The GRADIO Interface
with gr.Blocks(theme=gr.themes.Base()) as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            # Chatbot interface
            chatbot = gr.Chatbot(label=llm_chain.llm_name, value=[], elem_id="chatbot")
        # with gr.Column(scale=1):
        #   # Uploaded PDFs window
        with gr.Column(scale=1):
            with gr.Column():
                file_output = gr.File(label="Your PDFs")
            # PDF upload button
            with gr.Column():
                btn = gr.UploadButton(
                    "📁 Upload a PDF(s)", file_types=[".pdf"], file_count="multiple"
                )
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            # Ask question input field
            txt = gr.Text(show_label=False, placeholder="Enter question")
        submit_btn = gr.Button("➡️ Ask")

    btn.upload(fn=llm_chain.load_docs, inputs=[btn], outputs=[file_output])

    # Gradio EVENTS
    # Event handler for submitting text question and generating response
    submit_btn.click(
        fn=llm_chain.response, inputs=[chatbot, txt], outputs=[chatbot, txt]
    )

if __name__ == "__main__":
    demo.launch()  # launch app
