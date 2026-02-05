/**
 * ACT Steering Demo - Frontend JavaScript
 */

class ACTDemo {
    constructor() {
        this.initElements();
        this.bindEvents();
        this.loadState();
    }

    initElements() {
        // Chat elements
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');

        // Model status
        this.modelStatus = document.getElementById('modelStatus');
        this.modelStatusText = document.getElementById('modelStatusText');

        // EPA bars
        this.eBar = document.getElementById('eBar');
        this.pBar = document.getElementById('pBar');
        this.aBar = document.getElementById('aBar');
        this.eValue = document.getElementById('eValue');
        this.pValue = document.getElementById('pValue');
        this.aValue = document.getElementById('aValue');

        // Metrics
        this.deflectionValue = document.getElementById('deflectionValue');
        this.avgDeflectionValue = document.getElementById('avgDeflectionValue');
        this.turnCount = document.getElementById('turnCount');
        this.errorValue = document.getElementById('errorValue');

        // Identity settings
        this.agentIdentity = document.getElementById('agentIdentity');
        this.userIdentity = document.getElementById('userIdentity');

        // Generation settings
        this.maxNewTokens = document.getElementById('maxNewTokens');
        this.maxTokensValue = document.getElementById('maxTokensValue');
        this.temperature = document.getElementById('temperature');
        this.temperatureValue = document.getElementById('temperatureValue');
        this.topP = document.getElementById('topP');
        this.topPValue = document.getElementById('topPValue');
        this.steeringCoeff = document.getElementById('steeringCoeff');
        this.steeringCoeffValue = document.getElementById('steeringCoeffValue');

        // Controller settings
        this.controllerEnabled = document.getElementById('controllerEnabled');
        this.contextMode = document.getElementById('contextMode');
        this.windowSizeGroup = document.getElementById('windowSizeGroup');
        this.windowSize = document.getElementById('windowSize');
        this.useDecay = document.getElementById('useDecay');
        this.decayRate = document.getElementById('decayRate');
        this.decayRateValue = document.getElementById('decayRateValue');
        this.applySettings = document.getElementById('applySettings');
        this.resetConversation = document.getElementById('resetConversation');
    }

    bindEvents() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        // Settings
        this.applySettings.addEventListener('click', () => this.applyConfig());
        this.resetConversation.addEventListener('click', () => this.resetChat());

        // Context mode toggle
        this.contextMode.addEventListener('change', () => {
            this.windowSizeGroup.style.display =
                this.contextMode.value === 'history' ? 'block' : 'none';
        });

        // Slider value displays
        this.decayRate.addEventListener('input', () => {
            this.decayRateValue.textContent = this.decayRate.value;
        });

        this.maxNewTokens.addEventListener('input', () => {
            this.maxTokensValue.textContent = this.maxNewTokens.value;
        });

        this.temperature.addEventListener('input', () => {
            this.temperatureValue.textContent = this.temperature.value;
        });

        this.topP.addEventListener('input', () => {
            this.topPValue.textContent = this.topP.value;
        });

        this.steeringCoeff.addEventListener('input', () => {
            this.steeringCoeffValue.textContent = this.steeringCoeff.value;
        });
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // Add user message to chat
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.messageInput.disabled = true;
        this.sendBtn.disabled = true;

        // Show loading indicator
        const loadingMsg = this.addMessage('Generating response...', 'loading');

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });

            // Remove loading message
            loadingMsg.remove();

            const data = await response.json();

            if (response.ok) {
                // Add assistant message
                this.addMessage(data.response, 'assistant', data.actual_epa);

                // Update EPA display
                this.updateEPABars(data.actual_epa);

                // Update metrics
                this.updateMetrics(data.metrics);
            } else {
                this.addMessage(`Error: ${data.detail}`, 'system');
            }
        } catch (error) {
            loadingMsg.remove();
            this.addMessage(`Error: ${error.message}`, 'system');
        }

        this.messageInput.disabled = false;
        this.sendBtn.disabled = false;
        this.messageInput.focus();
    }

    addMessage(content, role, epa = null) {
        const div = document.createElement('div');
        div.className = `message ${role}`;

        let html = `<p>${this.escapeHtml(content)}</p>`;

        if (epa && role === 'assistant') {
            html += `<div class="message-epa">EPA: E=${epa.e.toFixed(2)}, P=${epa.p.toFixed(2)}, A=${epa.a.toFixed(2)}</div>`;
        }

        div.innerHTML = html;
        this.chatMessages.appendChild(div);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        return div;
    }

    updateEPABars(epa) {
        // Convert EPA values (-4.3 to 4.3) to percentage (0-100)
        const toPercent = (val) => ((val + 4.3) / 8.6) * 100;
        const toWidth = (val) => Math.abs(val) / 4.3 * 50;

        // E bar
        const ePercent = toPercent(epa.e);
        const eLeft = epa.e >= 0 ? 50 : 50 - toWidth(epa.e);
        this.eBar.style.left = `${eLeft}%`;
        this.eBar.style.width = `${toWidth(epa.e)}%`;
        this.eValue.textContent = epa.e.toFixed(2);

        // P bar
        const pLeft = epa.p >= 0 ? 50 : 50 - toWidth(epa.p);
        this.pBar.style.left = `${pLeft}%`;
        this.pBar.style.width = `${toWidth(epa.p)}%`;
        this.pValue.textContent = epa.p.toFixed(2);

        // A bar
        const aLeft = epa.a >= 0 ? 50 : 50 - toWidth(epa.a);
        this.aBar.style.left = `${aLeft}%`;
        this.aBar.style.width = `${toWidth(epa.a)}%`;
        this.aValue.textContent = epa.a.toFixed(2);
    }

    updateMetrics(metrics) {
        if (metrics.total_deflection !== undefined) {
            this.deflectionValue.textContent = metrics.total_deflection.toFixed(2);
        }
        if (metrics.avg_deflection !== undefined) {
            this.avgDeflectionValue.textContent = metrics.avg_deflection.toFixed(2);
        }
        if (metrics.turn_count !== undefined) {
            this.turnCount.textContent = metrics.turn_count;
        }
        if (metrics.current_error !== undefined) {
            this.errorValue.textContent = metrics.current_error.toFixed(2);
        }
    }

    updateModelStatus(loaded, modelName) {
        const indicator = this.modelStatus.querySelector('.status-indicator');

        if (loaded) {
            indicator.className = 'status-indicator loaded';
            this.modelStatusText.textContent = `Model: ${modelName || 'Loaded'}`;
            this.sendBtn.disabled = false;
            this.messageInput.disabled = false;
        } else {
            indicator.className = 'status-indicator loading';
            this.modelStatusText.textContent = 'Loading model...';
            this.sendBtn.disabled = true;
            this.messageInput.disabled = true;
        }
    }

    async applyConfig() {
        const config = {
            identities: {
                agent_identity: this.agentIdentity.value,
                user_identity: this.userIdentity.value,
                agent_modifiers: [],
                user_modifiers: []
            },
            controller: {
                enabled: this.controllerEnabled.checked,
                context_mode: this.contextMode.value,
                window_size: parseInt(this.windowSize.value),
                use_decay: this.useDecay.checked,
                decay_rate: parseFloat(this.decayRate.value),
                kp: 1.0,
                ki: 0.1,
                kd: 0.05
            },
            generation: {
                max_new_tokens: parseInt(this.maxNewTokens.value),
                temperature: parseFloat(this.temperature.value),
                top_p: parseFloat(this.topP.value),
                steering_coefficient: parseFloat(this.steeringCoeff.value)
            }
        };

        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                this.addMessage('✓ Settings applied successfully', 'system');
            } else {
                const data = await response.json();
                this.addMessage(`Error: ${data.detail}`, 'system');
            }
        } catch (error) {
            this.addMessage(`Error: ${error.message}`, 'system');
        }
    }

    async resetChat() {
        try {
            const response = await fetch('/api/reset', { method: 'POST' });

            if (response.ok) {
                // Clear chat messages except system
                this.chatMessages.innerHTML = '';
                this.addMessage('Conversation reset. Start a new chat!', 'system');

                // Reset metrics
                this.deflectionValue.textContent = '0.00';
                this.avgDeflectionValue.textContent = '0.00';
                this.turnCount.textContent = '0';
                this.errorValue.textContent = '0.00';

                // Reset EPA bars
                this.updateEPABars({ e: 0, p: 0, a: 0 });
            }
        } catch (error) {
            this.addMessage(`Error: ${error.message}`, 'system');
        }
    }

    async loadState() {
        try {
            const response = await fetch('/api/state');
            if (response.ok) {
                const data = await response.json();

                // Update model status
                this.updateModelStatus(data.model_loaded, data.model_name);

                // Update identity settings from state
                if (data.agent) {
                    this.agentIdentity.value = data.agent.identity.split(' ').pop() || 'assistant';
                }
                if (data.user) {
                    this.userIdentity.value = data.user.identity.split(' ').pop() || 'person';
                }

                // Update generation settings
                if (data.generation) {
                    this.maxNewTokens.value = data.generation.max_new_tokens;
                    this.maxTokensValue.textContent = data.generation.max_new_tokens;
                    this.temperature.value = data.generation.temperature;
                    this.temperatureValue.textContent = data.generation.temperature;
                    this.topP.value = data.generation.top_p;
                    this.topPValue.textContent = data.generation.top_p;
                    this.steeringCoeff.value = data.generation.steering_coefficient;
                    this.steeringCoeffValue.textContent = data.generation.steering_coefficient;
                }

                // Update controller settings
                if (data.controller) {
                    this.controllerEnabled.checked = data.controller.enabled;
                    this.contextMode.value = data.controller.config.context_mode;
                    this.windowSize.value = data.controller.config.window_size;
                    this.useDecay.checked = data.controller.config.use_decay;
                    this.decayRate.value = data.controller.config.decay_rate;
                    this.decayRateValue.textContent = data.controller.config.decay_rate;

                    this.windowSizeGroup.style.display =
                        this.contextMode.value === 'history' ? 'block' : 'none';
                }

                if (data.metrics) {
                    this.updateMetrics(data.metrics);
                }
            }
        } catch (error) {
            console.error('Failed to load state:', error);
            // Keep trying until model is loaded
            setTimeout(() => this.loadState(), 2000);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    window.actDemo = new ACTDemo();
});
