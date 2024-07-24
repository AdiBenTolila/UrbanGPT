const selectDoc = document.getElementById('docSelect');
const messageInput = document.getElementById('messageInput');
const chatContainer = document.querySelector('.chat-container');
var messages_list = [];
/*
let selectedDoc;
fetch('/available_plans', {
    method: 'GET',
    headers: {
        'Content-Type': 'application/json'
    }
}).then(response => response.json())
.then(data => {
    data.forEach(doc => {
        const option = document.createElement('option');
        option.value = doc;
        option.textContent = doc;
        selectDoc.appendChild(option);
    });
})
.catch(error => console.error('Error:', error));
selectDoc.addEventListener('change', function() {
    if (selectDoc.value !== 'Select a document' && selectedDoc !== selectDoc.value) {
        selectedDoc = selectDoc.value;
        const userMessage = document.createElement('div');
        userMessage.className = 'chat-box user-message';
        userMessage.innerHTML = `<table><tr><td><span>Selected document: ${selectedDoc}</span></td><td><img src="static/user-icon.jpg"></td></tr></table>`;
        chatContainer.appendChild(userMessage);
    }
});
*/

function append_message(message, type) {
    const messageElement = document.createElement('div');
    if(type === 'human') {
        message = message.replace(/\n/g, '<br>');
        messageElement.className = 'chat-box user-message';
        messageElement.innerHTML = `<table>
                                        <tr>
                                            <td>
                                                <span>${message}</span>
                                            </td>
                                            <td style="text-align: end" class="icon-td">
                                                <img src="static/user-icon.jpg">
                                            </td>
                                        </tr>
                                    </table>`;
    } else if(type === 'ai') {
        message = message.replace(/\n/g, '<br>');
        messageElement.className = 'chat-box bot-message';
        messageElement.innerHTML = `<table>
                                        <tr>
                                            <td class="icon-td">
                                                <img src="static/bot_icon.png">
                                            </td>
                                            <td>
                                                <span>${message}</span>
                                            </td>
                                        </tr>
                                    </table>`;
    } else if(type === 'tool_call') {
        messageElement.className = 'chat-box source-message';
        messageElement.id = message.id;
        // messageElement.innerHTML = `<table id="table-${obj.id}"><tr><td class="icon-td"><img src="static/source_icon.png"></td><td><span>${message}</span></td></tr></table>`;
        messageElement.innerHTML = `<table id="table-${message.id}">
                                        <tr>
                                            <td class="icon-td">
                                                <img src="static/source_icon.png">
                                            </td>
                                            <td>
                                                <span>${message.name}(${Object.entries(message.args).map(([key, value]) => `${key}=${value}`).join(', <br/>')})</span>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td class="icon-td">
                                                <img src="static/source_icon.png">
                                            </td>
                                            <td id="result-${message.id}">
                                                <span class="loader"></span>
                                            </td>
                                    </table>`;
    }else if(type === 'tool_message') {
        console.log(message);
        const toolPlaceholder = document.getElementById(`result-${message.tool_call_id}`);
        toolPlaceholder.innerHTML = `<span>${message.result.replace(/\n/g, '<br>')}</span>`;
        return;
    }else {
        console.error('Invalid message type '+ type);
        return;
    }
    chatContainer.appendChild(messageElement);
}

function sendAgentMessage(){
    const message = messageInput.value.trim();
    if (message) {
        fetch("/submit_history", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ history: messages_list })
        }).then(response => response.json())
        .then(data => {
            const eventSource = new EventSource(`/ask_agent?question=${message}`);
            append_message(message, 'human');
            eventSource.onmessage = function(data) {
                handleAgentMessage(JSON.parse(data.data));
            };

            eventSource.onerror = function(event) {
                console.error('EventSource failed:', event);
                eventSource.close();
            };
            messageInput.value = '';
        }).catch(error => console.error('Error:', error));
    }
}

function sendMessage() {
    const message = messageInput.value.trim();
    if (message) {
        if (selectDoc.value === 'Select a document') {
            fetch('/ask_general', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: message })
            })
            .then(response => response.json())
            .then(data => displayResTable(data.columns, data.data))
            .catch(error => console.error('Error:', error));
        } else {
            fetch(`/ask/${selectedDoc}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: message })
            })
            .then(response => response.json())
            .then(data => displayBotMessage(data.message, data.sources))
            .catch(error => console.error('Error:', error));
        }
        messageInput.value = '';
    }
}

function handleAgentMessage(data) {
    messages_list.push(data);
    const { type, content, tool_calls, tool_call_id } = data;

    if (type === 'human') {
        return;
    } else if (type === 'ai') {
        if (content) {
            append_message(content, type);
        }
        tool_calls.forEach(tool_call => {
            append_message(tool_call, 'tool_call');
        });
    } else if (type === 'tool') {
        append_message({result: content, tool_call_id}, 'tool_message');
    }
}

function displayBotMessage(message, sources) {
    if (sources) {
        sources.forEach(source => {
            const sourceMessage = document.createElement('div');
            sourceMessage.className = 'chat-box source-message';
            sourceMessage.innerHTML = `<table  style="display:inline"><tr><td class="icon-td"><img src="static/source_icon.png" style="margin-right: 8px; border-radius: 50%; width: 40px;"></td><td><span>${sources[i]}</span></td></tr></table>`;
            chatContainer.appendChild(sourceMessage);
        });
    }
    const botMessage = document.createElement('div');
    botMessage.className = 'chat-box bot-message';
    botMessage.innerHTML = `<table><tr><td class="icon-td"><img src="static/bot_icon.png"></td><td><span>${message}</span></td></tr></table>`;
    chatContainer.appendChild(botMessage);
}

function displayResTable(cols, data) {
    const table = document.createElement('table');
    const headerRow = table.insertRow(-1);

    cols.forEach(col => {
        const th = document.createElement('th');
        th.innerText = col;
        headerRow.appendChild(th);
    });

    data.forEach(row => {
        const tr = table.insertRow(-1);
        cols.forEach(col => {
            const td = tr.insertCell(-1);
            td.innerText = row[col];
        });
    });

    const botMessage = document.createElement('div');
    botMessage.className = 'chat-box bot-message';
    botMessage.innerHTML = `<table><tr><td><img src="static/bot_icon.png"></td><td><span>Here are the results:</span></td></tr></table>`;
    botMessage.appendChild(table);
    chatContainer.appendChild(botMessage);
}

function cleanChat() {
    messages_list = [];
    chatContainer.innerHTML = '';
    fetch('/submit_history', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ history: messages_list })
    }).catch(error => console.error('Error:', error));
    append_message("ברוך הבא לUrbanGPT! תוכל לשאול אותי כל שאלה שתרצה", "ai");
}   

function onLoad() {
    const newChatBtn = document.getElementById('newChatBtn');
    const chatModeList = document.getElementById('chatModeList');
    const editSettingBtn = document.getElementById('editSettingBtn');
    const clearDataBtn = document.getElementById('clearDataBtn');
    const settingModal = document.getElementById('settingsModal');
    const closeModal = document.querySelector('.modal .close');
    const saveSettingBtn = document.getElementById('saveSettingsBtn');
    const chatList = document.getElementById('chatList');
    const systemMessageTextarea = document.getElementById('systemMessage');
    const modelSelect = document.getElementById('modelName');
    const numRetrievalDocsInput = document.getElementById('numRetrievalDocs');
    


    document.querySelectorAll(".navbar-link").forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const href = e.target.getAttribute('href');
            const section = e.target.getAttribute('data-section');
            document.querySelectorAll('.content-section').forEach(section => {
                section.classList.remove('active');
            });
            document.querySelector(`#${section}`).classList.add('active');
        });
    });

    fetch('/available_models', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json())
    .then(data => {
        data.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    })

    // Event listener for starting a new chat
    newChatBtn.addEventListener('click', function() {
        cleanChat();
    });

    clearDataBtn.addEventListener('click', function() {
        fetch('/clear_data').catch(error => console.error('Error:', error));
    });
    // Event listener for changing chat mode
    chatModeList.addEventListener('click', function(e) {
        if (e.target.tagName === 'LI') {
            const selectedMode = e.target.getAttribute('data-mode');
            // toggle active class on the selected mode
            chatModeList.querySelectorAll('li').forEach(li => {
                if (li.getAttribute('data-mode') === selectedMode) {
                    li.classList.add('active');
                } else {
                    li.classList.remove('active');
                }
            });
            
            // Handle chat mode change logic here
            console.log('Chat mode changed to:', selectedMode);
        }
    });

    // Event listener for opening the system message modal
    editSettingBtn.addEventListener('click', function() {
        settingModal.style.display = 'block';
        fetch('/get_attributes', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(response => response.json())
        .then(data => {
            console.log(data);
            systemMessageTextarea.value = data.system_message;
            modelSelect.value = data.model;
            numRetrievalDocsInput.value = data.num_chunks;
        })

    });
    
    
    // Event listener for closing the system message modal
    closeModal.addEventListener('click', function() {
        settingModal.style.display = 'none';
    });

    // Event listener for saving system message
    saveSettingBtn.addEventListener('click', function() {
        const systemMessage = systemMessageTextarea.value;
        const model = modelSelect.value;
        const numRetrievalDocs = numRetrievalDocsInput.value;
        fetch('/set_attributes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                system_message: systemMessage, 
                model: model, 
                num_chunks: numRetrievalDocs })
        }).catch(error => console.error('Error:', error));

        // Handle system message saving logic here
        settingModal.style.display = 'none';
    });

    // Load previous chats (this is a placeholder, replace with your actual implementation)
    const previousChats = ['Chat 1', 'Chat 2', 'Chat 3'];
    previousChats.forEach(chat => {
        const li = document.createElement('li');
        li.textContent = chat;
        li.addEventListener('click', function() {
            console.log('Loading chat:', chat);

            // Handle loading of the selected previous chat
        });
        chatList.appendChild(li);
    });
    
    // Close the modal when clicking outside of it
    window.addEventListener('click', function(event) {
        const settingModal = document.getElementById('settingModal');
        if (event.target === settingModal) {
            settingModal.style.display = 'none';
        }
    });
    
    // Event listener for sending a new chat message
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendAgentMessage();
        }
    });
    document.getElementById('uploadBtn').addEventListener('click', sendAgentMessage);



    append_message("ברוך הבא לUrbanGPT! תוכל לשאול אותי כל שאלה שתרצה", "ai");
}