import streamlit as st
import json
from datetime import datetime
from database.operations import execute_query, execute_insert, safe_db_operation
from config import init_client, MODEL

client = init_client()

def chat(user_input):
    if not user_input or not client:
        return

    st.session_state.generating = True

    # Add the user's message to the conversation
    st.session_state.conversation.append({"role": "user", "content": user_input})

    # Display the user's message
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        try:
            combined_context = create_combined_context()

            with st.spinner("Generating response..."):
                # Send the entire conversation history
                messages_to_send = truncate_conversation(st.session_state.conversation)

                for event in client.messages.create(
                        max_tokens=st.session_state.max_tokens,
                        temperature=st.session_state.temperature,
                        top_p=st.session_state.top_p,
                        system=combined_context,
                        messages=messages_to_send,
                        model=MODEL,
                        stream=True,
                ):
                    if st.session_state.generating:
                        if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                            full_response += event.delta.text
                            response_container.markdown(f"{full_response}▌")
                    else:
                        break

            response_container.markdown(full_response)

            # Add the assistant's response to the conversation history
            st.session_state.conversation.append({"role": "assistant", "content": full_response})

            # Save the updated chat state
            save_chat_state()

        except Exception as e:
            response_container.error(f"An error occurred: {e}")
            st.error(f"Error during API call: {e}")
        finally:
            st.session_state.generating = False

def create_combined_context():
    combined_context = f"{st.session_state.context}\n\n"

    # Include a summary of the last 10 chats
    conversation_summary = ""
    chat_count = 0
    total_tokens = 0
    max_tokens = 50000  # Increased token limit

    for msg in reversed(st.session_state.conversation):
        if chat_count >= 10:
            break

        role = msg['role']
        content = msg['content']

        # Simple compression: truncate long messages and remove extra whitespace
        if len(content) > 500:
            content = content[:497] + "..."
        content = ' '.join(content.split())  # Remove extra whitespace

        summary = f"{role.capitalize()}: {content}\n\n"
        summary_tokens = len(summary.encode('utf-8'))

        if total_tokens + summary_tokens > max_tokens:
            break

        conversation_summary = summary + conversation_summary
        total_tokens += summary_tokens
        chat_count += 1

    combined_context += f"Recent conversation history:\n{conversation_summary}"

    if st.session_state.document_content:
        # Compress document content if needed
        doc_content = st.session_state.document_content
        if len(doc_content) > 1000:
            doc_content = doc_content[:997] + "..."
        doc_content = ' '.join(doc_content.split())  # Remove extra whitespace
        combined_context += f"\n\nDocument Content:\n{doc_content}"

    # Ensure we don't exceed the max token limit
    if len(combined_context.encode('utf-8')) > max_tokens:
        combined_context = combined_context[:max_tokens].rsplit(' ', 1)[0] + "..."

    return combined_context

def truncate_conversation(conversation, max_messages=50):
    if len(conversation) > max_messages:
        truncated = conversation[-max_messages:]
        truncated[0]["content"] = f"[Earlier conversation truncated] ... {truncated[0]['content']}"
        return truncated
    return conversation

def save_chat_state():
    safe_db_operation(lambda cur: cur.execute(
        "DELETE FROM chat_state WHERE user_id = ?", (st.session_state.user_id,)
    ))
    safe_db_operation(lambda cur: cur.execute(
        "INSERT INTO chat_state (user_id, conversation, document_content, context) VALUES (?, ?, ?, ?)",
        (st.session_state.user_id, json.dumps(st.session_state.conversation),
         st.session_state.document_content, st.session_state.context)
    ))

def load_chat_state():
    result = execute_query("SELECT * FROM chat_state WHERE user_id = ?", (st.session_state.user_id,))
    if result:
        st.session_state.conversation = json.loads(result[0][2])
        st.session_state.document_content = result[0][3]
        st.session_state.context = result[0][4]
    else:
        st.session_state.conversation = []
        st.session_state.document_content = ""
        st.session_state.context = "You are a helpful assistant with tool calling capabilities. The user has access to the tool's outputs that you as a model cannot see. This could include text, images and more."

def save_conversation():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation_json = json.dumps(st.session_state.conversation)
    title = generate_conversation_title(conversation_json)
    execute_insert(
        "INSERT INTO conversations (user_id, conversation, timestamp, context, title) VALUES (?, ?, ?, ?, ?)",
        (st.session_state.user_id, conversation_json, timestamp, st.session_state.context, title)
    )

def generate_conversation_title(conversation):
    sample = json.loads(conversation)[:3]  # Take the first 3 messages as a sample
    sample_text = "\n".join([f"{msg['role']}: {msg['content'][:50]}..." for msg in sample])

    prompt = f"Generate a short, 1-2 word title for this conversation:\n\n{sample_text}\n\nTitle:"

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
            system="You are a helpful assistant that generates short, concise titles."
        )
        title = response.content[0].text.strip()
        return title
    except Exception as e:
        st.error(f"Error generating title: {e}")
        return "Untitled Chat"