const messageInput = document.getElementById('messageInput');
// const chatContainer = document.querySelector('.chat-container');
var messages_list = [];
var chats = {};
var currentChat = undefined;
var selectedChatMode = "agent";
var md2html_converter = new showdown.Converter()
md2html_converter.setOption('tables', true);

function append_message(message, type, conversation_id, message_id=undefined) {
    // html      = converter.makeHtml(text);

    if (conversation_id === undefined) {
        conversation_id = "new"
    }
    
    const chatContainer = document.querySelector(`#chat-${conversation_id}`).querySelector('.chat-body');
    const messageElement = document.createElement('div');
    if(type === 'human') {
        message = md2html_converter.makeHtml(message);
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
        id_str = message_id ? `id="${message_id}"` : '';
        // message = message.replace(/\n/g, '<br>');
        message = md2html_converter.makeHtml(message);
        messageElement.className = 'chat-box bot-message';
        messageElement.innerHTML = `<table>
                                        <tr>
                                            <td class="icon-td">
                                                <img src="static/bot_icon.png">
                                            </td>
                                            <td>
                                                <span ${id_str}>${message}</span>
                                            </td>
                                        </tr>
                                    </table>`;
        
    } else if(type === 'tool_call') {
        messageElement.className = 'chat-box source-message';
        messageElement.id = message.id;
        let message_text = `${message.name}(${Object.entries(message.args).map(([key, value]) => `${key}=${value}`).join(', \n')})`
        // escape _ in message_text
        message_text = message_text.replace(/_/g, '\\_');
        let message_text_html = md2html_converter.makeHtml(message_text);
        // messageElement.innerHTML = `<table id="table-${obj.id}"><tr><td class="icon-td"><img src="static/source_icon.png"></td><td><span>${message}</span></td></tr></table>`;
        messageElement.innerHTML = `<table id="table-${message.id}">
                                        <tr>
                                            <td>
                                                <span>${message_text_html}</span>
                                            </td>
                                            <td class="icon-td">
                                                <img src="static/source_icon.png">
                                            </td>

                                        </tr>
                                        <tr>
                                            <td id="result-${message.id}">
                                                <span class="loader"></span>
                                            </td>
                                            <td class="icon-td">
                                                <img src="static/source_icon.png">
                                            </td>

                                        </tr>
                                    </table>`;
    }else if(type === 'tool_message') {
        const md_message = md2html_converter.makeHtml(message.result.replace(/_/g, '\\_'));
        const toolPlaceholder = document.querySelector(`#chat-${conversation_id} #result-${message.tool_call_id}`);
        toolPlaceholder.innerHTML = `<span>${md_message}</span>`;
        return;
    }else if(type === 'info') {
        messageElement.className = 'chat-box info-message';
        messageElement.innerHTML = `<h3>${message}</h3>`;
    }else {
        console.error('Invalid message type '+ type);
        return;
    }
    chatContainer.appendChild(messageElement);
}

function create_new_chat(chat_id, title, from_new=false) {
    if (title === undefined) {
        title = 'צ\'אט חדש';
    }
    let chat = document.createElement('div');
    if (from_new){
        chat = document.getElementById('chat-new')
        chat.id = `chat-${chat_id}`
        chat.querySelector('chat-header').innerHTML = `<h3 id=">${title}</h3>`

        const newChat = document.createElement('div');
        newChat.className = 'chat-container';
        newChat.id = `chat-new`;
        newChat.innerHTML = `<div class="chat-header"><h3>צ'אט חדש</h3></div><div class="chat-body"></div>`;
        document.getElementById('chat-list').appendChild(newChat);
    }else{
        chat.className = 'chat-container';
        chat.id = `chat-${chat_id}`;
        chat.innerHTML = `<div class="chat-header"><h3>${title}</h3></div><div class="chat-body"></div>`;
        document.getElementById('chat-list').appendChild(chat);
    }

    const li = document.createElement('li');
    li.textContent = title;
    li.setAttribute('data-conversation-id', chat_id);
    li.addEventListener('click', function(e) {
        display_chat(e.target.getAttribute('data-conversation-id'))
    });
    const chatList = document.getElementById('chatList');
    chatList.insertBefore(li, chatList.firstChild);
    return chat;
}

function display_chat(chat_id) {
    console.log(`Displaying chat ${chat_id}`);
    if (chat_id === 'new') {
        currentChat = undefined;
    }else {
        currentChat = chat_id;
    }
    const chats = document.querySelectorAll('.chat-container');
    chats.forEach(chat => {
        chat.classList.remove('active');
    });
    let chat = document.getElementById(`chat-${chat_id}`);
    if (!chat) {
        console.error('Chat not found');
        chat = create_new_chat(chat_id);
    }
    chat.classList.add('active');

    const chatList = document.getElementById('chatList').querySelectorAll('li');
    const chatLI = document.querySelector(`#chatList li[data-conversation-id="${chat_id}"]`);
    chatList.forEach(li => {
        li.classList.remove('active');
    });
    if (chatLI) {
        chatLI.classList.add('active');
    }
}

function sendMessage() {
    const message = messageInput.value.trim();
    messageInput.value = '';
    const conversation_id = currentChat;

    if (message) {
        append_message(message, 'human', conversation_id);
        const eventSource = new EventSource(`/chat?question=${message}&conversation_id=${conversation_id}`);
        eventSource.onmessage = function(data) {
            handleMessage(JSON.parse(data.data));
        };

        eventSource.onerror = function(event) {
            console.error('EventSource failed:', event);
            eventSource.close();
        };
        messageInput.value = '';
    }

}

function handleMessage(data) {
    const {conversation_id, title, message_id, message, mode} = data;
    if (currentChat === undefined) {
        currentChat = conversation_id;
    }
    if (title !== undefined) {
        // update chat title
        const chat_title = document.querySelector(`#chat-${conversation_id} .chat-header h3`);
        const chat_link = document.querySelector(`#chatList li[data-conversation-id="${conversation_id}"]`);
        chat_title.innerText = title;
        chat_link.innerText = title;        
    }
    const { type, content, tool_calls, tool_call_id} = JSON.parse(message);
    if (type === 'human') {
        // append_message(content, type, conversation_id);
    } else if (type === 'ai') {
        if (content) {
            append_message(content, type, conversation_id, message_id);
        }
        tool_calls.forEach(tool_call => {
            append_message(tool_call, 'tool_call', conversation_id);
        });
    } else if (type === 'tool') {
        append_message({result: content, tool_call_id}, 'tool_message', conversation_id);
    }else if (type === 'AIMessageChunk') {
        const messageElement = document.querySelector(`#chat-${conversation_id} #${message_id}`);
        if (messageElement) {
            let txt_content = messageElement.getAttribute('data-content') || '';
            txt_content += content;
            messageElement.setAttribute('data-content', txt_content);
            messageElement.innerHTML = md2html_converter.makeHtml(txt_content);
        }else {
            append_message(content, 'ai', conversation_id, message_id);
        }
    }
}

function displayBotMessage(message, sources) {
}

function createResTable(cols, data) {
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
    return table
}

function newChat() {
    currentChat = undefined;
    display_chat('new');
}   

// function filterCheckboxes() {
//     var input, filter, div, labels, i, txtValue, selectAll;
//     input = document.getElementById('searchBox');
//     filter = input.value.toUpperCase();
//     div = document.getElementById('documentsCheckboxes').querySelector('.checkbox-container');
//     labels = div.getElementsByTagName('label');
//     selectAll = document.getElementById('selectAll');

//     for (i = 0; i < labels.length; i++) {
//         txtValue = labels[i].textContent || labels[i].innerText;
//         console.log(txtValue + ' ' + filter);
//         if (txtValue.toUpperCase().indexOf(filter) > -1) {
//             labels[i].parentElement.style.display = "";
//         } else {
//             labels[i].parentElement.style.display = "none";
//         }
//     }
//     selectAll.checked = false;
// }

function toggleSelectAll() {
    const checkboxes = document.querySelectorAll('.checkbox-button');
    const selectAll = document.getElementById('selectAll');
    checkboxes.forEach(checkbox => {
        if (checkbox.parentElement.style.display !== 'none') {
            checkbox.checked = selectAll.checked;
        }
    });
}
let filters = [];

function filterCheckboxes() {
    const checkboxes = document.querySelectorAll('.checkbox-button');
    const selectAll = document.getElementById('selectAll');

    checkboxes.forEach(checkbox => {
        let match = true;

        filters.forEach(filter => {
            const { property, type, criteria } = filter;
            const value = checkbox.getAttribute(`data-${property}`);
            if (type === 'INTEGER' || type === 'FLOAT') {
                const min = parseFloat(criteria.min);
                const max = parseFloat(criteria.max);
                const numValue = parseFloat(value);

                if (isNaN(numValue) || (min && numValue < min) || (max && numValue > max)) {
                    match = false;
                }
            } else if (type === 'TEXT') {
                if (!value || !value.toLowerCase().includes(criteria.toLowerCase())) {
                    match = false;
                }
            } else if (type === 'BOOLEAN') {
                console.log(value, criteria);
                if (value.toLowerCase() !== criteria) {
                    match = false;
                }
            }else if (type === 'TIMESTAMP') {
                const min = new Date(criteria.min);
                const max = new Date(criteria.max);
                const dateValue = new Date(value);
                if ((min && dateValue < min) || (max && dateValue > max)) {
                    match = false;
                }
            }
        });

        checkbox.parentElement.style.display = match ? 'block' : 'none';
    });
    // selectAll is checked if all checkboxes are checked
    selectAll.checked = [...checkboxes].filter(checkbox => checkbox.parentElement.style.display !== 'none').every(checkbox => checkbox.checked);
}

function updateCriteriaInput() {
    const propertySelect = document.getElementById('property');
    const selectedOption = propertySelect.options[propertySelect.selectedIndex];
    const type = selectedOption.getAttribute('data-type');
    const criteriaInputContainer = document.getElementById('criteriaInputContainer');

    criteriaInputContainer.innerHTML = '';

    if (type === 'INTEGER' || type === 'FLOAT') {
        criteriaInputContainer.innerHTML = `
            <div class="form-group">
                <label for="minCriteria">מינימום:</label>
                <input type="number" id="minCriteria" class="form-control" placeholder="הזן ערך מינימלי..">
            </div>
            <div class="form-group">
                <label for="maxCriteria">מקסימום:</label>
                <input type="number" id="maxCriteria" class="form-control" placeholder="הזן ערך מקסימלי..">
            </div>
        `;
    } else if (type === 'TEXT') {
        criteriaInputContainer.innerHTML = `
            <div class="form-group">
                <label for="criteria">קריטריונים:</label>
                <input type="text" id="criteria" class="form-control" placeholder="הזן קריטריונים..">
            </div>
        `;
    } else if (type === 'BOOLEAN') {
        criteriaInputContainer.innerHTML = `
            <div class="form-group">
                <label for="criteria">קריטריונים:</label>
                <select id="criteria" class="form-control">
                    <option value="true">כן</option>
                    <option value="false">לא</option>
                </select>
            </div>
        `;
    } else if (type === 'TIMESTAMP') {
        criteriaInputContainer.innerHTML = `
            <div class="form-group">
                <label for="minCriteria">מינימום:</label>
                <input type="date" id="minCriteria" class="form-control">
            </div>
            <div class="form-group">
                <label for="maxCriteria">מקסימום:</label>
                <input type="date" id="maxCriteria" class="form-control">
            </div>
        `;
    } else {
        console.error('Invalid type');
    }
}

function addFilter() {
    const property = document.getElementById('property').value;
    const type = document.getElementById('property').selectedOptions[0].getAttribute('data-type');
    let criteria;

    if (type === 'INTEGER' || type === 'FLOAT' || type === 'TIMESTAMP') {
        const min = document.getElementById('minCriteria').value;
        const max = document.getElementById('maxCriteria').value;
        criteria = { min, max };
    } else if (type === 'TEXT' || type === 'BOOLEAN') {
        criteria = document.getElementById('criteria').value;
    }else {
        console.error('Invalid type');
    }


    if (property && criteria) {
        filters.push({ property, type, criteria });
        updateActiveFilters();
        filterCheckboxes();
        document.getElementById('filterModal').style.display = 'none';
    }
}

function updateActiveFilters() {
    const activeFiltersContainer = document.getElementById('activeFilters');
    activeFiltersContainer.innerHTML = '';

    filters.forEach((filter, index) => {
        let filterText;
        if (filter.type === 'INTEGER' || filter.type === 'FLOAT' || filter.type === 'TIMESTAMP') {
            // filterText = `${filter.property}: Min=${filter.criteria.min}, Max=${filter.criteria.max}`;
            filterText = `${filter.criteria.min} ≤ ${filter.property} ≤ ${filter.criteria.max}`;
        } else if (filter.type === 'TEXT') {
            // filterText = `${filter.property} כולל ${filter.criteria}`;
            filterText = `${filter.criteria} ⊆ ${filter.property}`;
        }
        else if (filter.type === 'BOOLEAN') {
            filterText = `${filter.property} = ${filter.criteria}`;
        }else {
            console.error('Invalid type');
        }
        const filterElement = document.createElement('div');
        filterElement.className = 'filter';
        filterElement.innerHTML = `
            <span type="button" class="filter-close" onclick="removeFilter(${index})">&times;</span>
            ${filterText}
        `;
        activeFiltersContainer.appendChild(filterElement);
    });
}

function removeFilter(index) {
    filters.splice(index, 1);
    updateActiveFilters();
    filterCheckboxes();
}

function initNewChatModal(){
    // set default attributes
    fetch('/get_attributes', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json())
    .then(data => {
        document.getElementById('chatSystemMessage').value = data.system_message;
        document.getElementById('chatModelName').value = data.model;
        document.getElementById('chatDocLLMName').value = data.doc_llm;
        document.getElementById('chatNumRetrievalDocs').value = data.num_chunks;
    })
}

function initSettingsModal(){
    fetch('/get_attributes', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json())
    .then(data => {    
        document.getElementById('systemMessage').value = data.system_message;
        document.getElementById('modelName').value = data.model;
        document.getElementById('docLLMName').value = data.doc_llm;
        document.getElementById('numRetrievalDocs').value = data.num_chunks;
    })
}

function setElementValue(elementId, value) {
    console.log(elementId, value);
    document.getElementById(elementId).innerHTML = value;
}
function editRow(docId) {
    const row = document.getElementById(`row-${docId}`);
    row.querySelectorAll('.editable-cell').forEach(cell => {
        const column = cell.getAttribute('data-column');
        if (column === 'id') {
            return;
        }
        const cellValue = cell.textContent.trim()
        // escape cell values
        const cellValueEscaped = cellValue.replace(/"/g, '&quot;');
        cell.innerHTML = `<input type="text" class="editable-input" value="${cellValueEscaped}" data-column="${column}">`;
        cell.classList.add('editing');
    });
    document.getElementById(`save-btn-${docId}`).classList.remove('d-none');
    document.getElementById(`edit-btn-${docId}`).classList.add('d-none');
}

function saveRow(docId) {
    const row = document.getElementById(`row-${docId}`);
    const updatedData = {};
    row.querySelectorAll('.editable-input').forEach(input => {
        const column = input.getAttribute('data-column');
        updatedData[column] = input.value.trim();
        const cell = input.parentElement;
        cell.innerHTML = input.value.trim();
        cell.classList.remove('editing');
    });

    // Send AJAX request to update the document
    fetch(`/document/${docId}`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(updatedData)
    }).then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Document updated successfully');
        } else {
            console.error('Failed to update document');
        }
    }).catch(error => console.error('Error:', error));
    document.getElementById(`save-btn-${docId}`).classList.add('d-none');
    document.getElementById(`edit-btn-${docId}`).classList.remove('d-none');
}

function deleteRow(docId) {
    const row = document.getElementById(`row-${docId}`);
    row.remove();
    // Send AJAX request to delete the document
    fetch(`/document/${docId}`, {
        method: 'DELETE'
    }).then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Document deleted successfully');
        } else {
            console.error('Failed to delete document');
        }
    }).catch(error => console.error('Error:', error));
}

function downloadDocument(docId) {
    fetch(`/document`, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({doc_id: docId})
    }).then(response => response.blob())
}

function addColumn(col_name, col_desc, col_type, model, num_docs, is_full_doc) {
    fetch('/column', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({name: col_name, description: col_desc, type: col_type, model: model, num_retrieval_docs: num_docs, is_full_doc: is_full_doc})
    }).then(response => response.json())
    .then(data => {
        if (data.status === 'ok') {
            $('#addColumnModal').css('display', 'none');
            console.log('Column added successfully');
            location.reload();
            // const table = document.getElementById('documentsTable');
            // const header = table.querySelector('thead tr');
            // const newCol = document.createElement('th');
            // newCol.innerText = col_name;
            // del_col_btn = document.createElement('button');
            // del_col_btn.className = 'btn btn-danger';
            // del_col_btn.setAttribute('data-toggle', 'tooltip');
            // del_col_btn.setAttribute('title', 'Delete Column');
            // del_col_btn.innerHTML = '<i class="fas fa-trash"></i>';
            // del_col_btn.addEventListener('click', function() {
            //     deleteColumn(col_name);
            // });
            // newCol.appendChild(del_col_btn);

            // // add new column at the end of the table, before the last column
            // header.insertBefore(newCol, header.children[header.children.length - 1]);
            // data.data.forEach(item => {
            //     const escaped_id = CSS.escape(item.id);
            //     const row = table.querySelector(`#row-${escaped_id}`);
            //     if (!row) {
            //         console.error(`Row with id ${escaped_id} not found`);
            //         return;
            //     }
            //     // <td class="editable-cell" data-column="{{ column }}">{{ doc[column] }}</td>
            //     const cell = document.createElement('td');
            //     cell.className = 'editable-cell';
            //     cell.setAttribute('data-column', col_name);
            //     cell.setAttribute('data-type', col_type);

            //     cell.innerText = item.answer;
            //     row.insertBefore(cell, row.children[row.children.length - 1]);
            // });
            // const rows = table.querySelectorAll('tbody tr');
            // rows.forEach(row => {
            //     const cell = document.createElement('td');
            //     cell.innerText = '';
            //     row.insertBefore(cell, row.children[row.children.length - 1]);
            // });
        } else {
            console.error('Failed to add column');
        }
    }).catch(error => console.error('Error:', error));
}

function deleteColumn(col_name) {
    fetch('/column', {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({name: col_name})
    }).then(response => response.json())
    .then(data => {
        if (data.status === 'ok') {
            console.log('Column deleted successfully');
            const table = document.getElementById('documentsTable');
            const header = table.querySelector('thead tr');
            const colIndex = [...header.children].findIndex(col => col.innerText === col_name);
            header.removeChild(header.children[colIndex]);
            const rows = table.querySelectorAll('tbody tr');
            rows.forEach(row => {
                row.removeChild(row.children[colIndex]);
            });
        } else {
            console.error('Failed to delete column');
        }
    }).catch(error => console.error('Error:', error));
}

function onLoad() {    
    // Event listener for opening modal
    const modalBtns = document.querySelectorAll('[data-toggle="modal"]');
    for(const btn of modalBtns){
        btn.addEventListener('click', function() {
            const modal = document.querySelector(btn.getAttribute('data-target'));
            // call modal callback function
            const callback_str = modal.getAttribute('data-callback');
            if (callback_str) {
                // evaluate callback function
                eval(callback_str);
            }
            modal.style.display = 'block';
        });
    }
    // Event listener for closing modal
    const closeModals = document.querySelectorAll('.modal .close');
    for(const closeModal of closeModals){
        closeModal.addEventListener('click', function(e) {
            const modal_id = e.target.parentElement.parentElement.parentElement.id;
            document.getElementById(modal_id).style.display = 'none';
        });
    }
    
    // Close the modal when clicking outside of it
    window.addEventListener('click', function(event) {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        });
    });
    
}

function onLoadIndex() {
    onLoad();
    // on chat type change show/hide retrieval fields in create new chat modal
    const modeSelect = document.getElementById('chatType');

    modeSelect.addEventListener('change', function() {
        const selectedMode = modeSelect.value;
        const retrieval_fields = document.getElementById('retrievalFields');
        if (selectedMode === 'agent') {
            retrieval_fields.style.display = 'block';
        } else {
            retrieval_fields.style.display = 'none';
        }
    });

    // Load previous chats
    fetch('/get_conversations').then(response => response.json())
    .then(data => {
        data.forEach(conversation => {
            const chat = create_new_chat(conversation.id, conversation.title, false);
            chats[conversation.id] = {messages: conversation.messages, title: conversation.title};
            for (const message of conversation.messages) {
                const { type, content, tool_calls, tool_call_id } = JSON.parse(message);
    
                if (type === 'ai') {
                    if (content) {
                        append_message(content, type, conversation.id);
                    }
                    tool_calls.forEach(tool_call => {
                        append_message(tool_call, 'tool_call', conversation.id);
                    });
                } else if (type === 'tool') {
                    append_message({result: content, tool_call_id}, 'tool_message', conversation.id);
                }else if (type === 'human') {
                    append_message(content, type, conversation.id);
                }
            }
        });
    })

    // Event listener for starting a new chat
    document.getElementById('createNewChat').addEventListener('click', function() {
        const selectedMode = document.getElementById('chatType').value;
        const selectedDocs = [...document.getElementById("documentsCheckboxes").querySelectorAll('input:checked')].map(doc => doc.value);
        const systemMessage = document.getElementById('chatSystemMessage').value;
        const model = document.getElementById('chatModelName').value;
        const docLLM = document.getElementById('chatDocLLMName').value;
        const numRetrievalDocs = document.getElementById('chatNumRetrievalDocs').value;

        document.getElementById('newChatModal').style.display = 'none';
        // TODO create new chat and show it
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                mode: selectedMode,
                documents: selectedDocs,
                system_message: systemMessage,
                model: model,
                num_chunks: numRetrievalDocs,
                doc_llm: docLLM
            })
        }).then(response => response.json())
        .then(data => {
            const {conversation_id, title} = data;
            // chats[conversation_id] = {messages: [], title: title, mode: selectedMode};
            const chat = create_new_chat(conversation_id, title);
            display_chat(conversation_id);
            currentChat = conversation_id;
        }).catch(error => console.error('Error:', error));
    });

    // Event listener for saving settings
    document.getElementById('saveSettingsBtn').addEventListener('click', function() {
        const systemMessage = document.getElementById('systemMessage').value;
        const model = document.getElementById('modelName').value;
        const numRetrievalDocs = document.getElementById('numRetrievalDocs').value;
        const docLLM = document.getElementById('docLLMName').value;
        fetch('/set_attributes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                system_message: systemMessage, 
                model: model, 
                num_chunks: numRetrievalDocs,
                doc_llm:  docLLM
                })
        }).catch(error => console.error('Error:', error));

        // Handle system message saving logic here
        settingModal.style.display = 'none';
    });
    
    const clearDataBtn = document.getElementById('clearDataBtn');
    clearDataBtn.addEventListener('click', function() {
        fetch('/clear_data').catch(error => console.error('Error:', error));
    });


    // Event listener for sending a new chat message
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    document.getElementById('uploadBtn').addEventListener('click', sendMessage);
    updateCriteriaInput();

    // append_message("ברוך הבא לUrbanGPT! תוכל לשאול אותי כל שאלה שתרצה", "info");

}

function onLoadDashboard() {
    onLoad();
    $('[data-toggle="tooltip"]').tooltip();
    $('[data-toggle2="tooltip"').tooltip();
    $('#addColumnForm').submit(function(e) {
        e.preventDefault();
        $('#addColumnForm').find('button[type="submit"]').html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...');

        const col_name = document.getElementById('column_name');
        const col_desc = document.getElementById('column_description');
        const col_type = document.getElementById('column_type');    
        const model = document.getElementById('model_name');
        const num_docs = document.getElementById('num_retrieval_docs');
        const is_full_doc = document.getElementById('is_full_doc');

        addColumn(col_name.value, col_desc.value, col_type.value, model.value, num_docs.value, is_full_doc.checked);
        col_name.value = '';
        col_desc.value = '';
        col_type.value = 'text';
        model.value = '';
        num_docs.value = '';
        is_full_doc.checked = false;
    });
    $('#uploadDocumentForm').submit(function(e) {
        e.preventDefault();
        $('#uploadDocumentForm').find('button[type="submit"]').html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...');
        const formData = new FormData(this);
        fetch('/document', {
            method: 'POST',
            body: formData,
            enctype: 'multipart/form-data'
        }).then(response => response.json())
        .then(data => {
            // show spinner
            $('#uploadDocumentModal').css('display', 'none');
            if (data.success) {
                location.reload();

                // console.log('Document added successfully');
                // const table = document.getElementById('documentsTable');
                // const newRow = table.insertRow(-1);
                // newRow.id = `row-${CSS.escape(data.doc_id)}`;
                // // TODO: add document to table
                // const columns = [...table.querySelector('thead tr').children].slice(0, -1);
                // const col_names = columns.map(col => col.innerText.trim());
                // const col_types = columns.map(col => col.getAttribute('data-type'));
                // for (let i = 0; i < col_names.length; i++) {
                //     const cell = newRow.insertCell(-1);
                //     if (col_types[i] === 'TIMESTAMP') {
                        
                //         cell.innerText = new Date(data.data[col_names[i]]/1000).toLocaleString();
                //     }else {
                //         cell.innerText = data.data[col_names[i]];
                //     }
                //     cell.className = 'editable-cell';
                //     cell.setAttribute('data-column', col_names[i]);
                //     cell.setAttribute('data-type', col_types[i]);
                // }                
            } else {
                console.error('Failed to add document');
            }
        }).catch(error => console.error('Error:', error));
    });
}