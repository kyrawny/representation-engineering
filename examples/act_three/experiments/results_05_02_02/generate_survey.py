import json
import os

input_file = "psytoolkit_examples.json"
output_file = "survey.txt"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

survey_lines = []

# Scale definition
survey_lines.append("scale: agree_7")
survey_lines.append("- Disagree strongly")
survey_lines.append("- Disagree")
survey_lines.append("- Disagree somewhat")
survey_lines.append("- Neither agree nor disagree")
survey_lines.append("- Agree somewhat")
survey_lines.append("- Agree")
survey_lines.append("- Agree strongly")
survey_lines.append("")

# Demographics
survey_lines.append("l: age")
survey_lines.append("t: textline")
survey_lines.append("q: How old are you?")
survey_lines.append("- {min=18,max=100} Enter your age:")
survey_lines.append("")

survey_lines.append("l: gender")
survey_lines.append("t: radio")
survey_lines.append("q: What is your gender?")
survey_lines.append("- Male")
survey_lines.append("- Female")
survey_lines.append("- Non-binary")
survey_lines.append("- Prefer to self-describe")
survey_lines.append("")

survey_lines.append("l: gender_other")
survey_lines.append("t: textline")
survey_lines.append("c: if $gender == 4")
survey_lines.append("q: Please specify your gender:")
survey_lines.append("- Enter your gender:")
survey_lines.append("")

survey_lines.append("random: begin")
survey_lines.append("")

for trial in data['trials']:
    scenario_id = trial['scenario_id']
    scenario_text = trial['scenario_text']
    agent_term = trial['agent_term']
    user_term = trial['user_term']
    
    conditions = trial['conditions']
    
    responses = []
    for cond_name, cond_data in conditions.items():
        text = cond_data['text']
        # remove LLM artifacts
        text = text.replace('end_header_id|>', '').strip()
        # replace newlines with space to avoid breaking psytoolkit item formatting
        text = text.replace('\n', ' ').replace('\r', '')
        # remove double quotes if necessary, or just escape them
        # simple way: replace double quotes with single quotes for safety in text
        text = text.replace('"', "'")
        # clean multiple spaces
        text = ' '.join(text.split())
        responses.append((cond_name, text))
        
    # Rank question
    survey_lines.append(f"l: rank_{scenario_id}")
    survey_lines.append("t: rank")
    survey_lines.append("o: random")
    survey_lines.append(f"q: Imagine a conversation between a {user_term} and a {agent_term}. The {user_term} says: \"{scenario_text}\". Order the following responses from the {agent_term}, starting with the one that feels the most socially appropriate.")
    for cond_name, text in responses:
        survey_lines.append(f"- {text}")
    survey_lines.append("")
    
    # Likert questions
    for cond_name, text in responses:
        survey_lines.append(f"l: likert_{scenario_id}_{cond_name}")
        survey_lines.append(f"q: Regarding the response: \"{text}\" - How much do you agree with each of the following statements?")
        survey_lines.append("t: scale agree_7")
        survey_lines.append("o: random")
        survey_lines.append("- This response is socially appropriate.")
        survey_lines.append("- This response is natural.")
        survey_lines.append("- This response is sycophantic (excessively praising or flattering someone in order to gain their approval or an advantage).")
        survey_lines.append("")

survey_lines.append("random: end")
survey_lines.append("")

survey_lines.append("l: finalquestion")
survey_lines.append("t: radio")
survey_lines.append("q: Did you enjoy the questionnaire?")
survey_lines.append("- yes")
survey_lines.append("- no")
survey_lines.append("")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(survey_lines))

print("Survey generated successfully to survey.txt")
