def build_multilingual_prompt(query, lang, domain):
    """TODO: Add description."""
    prompt_templates = {
        'en': {
            'retail': "Respond to this retail customer in English: {query}",
            'tech': "Provide technical support in English: {query}"
        },
        'es': {
            'retail': "Responde a este cliente de retail en espaol: {query}",
            'tech': "Proporciona soporte tcnico en espaol: {query}"
        },
        'ja': {
            'retail': ": {query}",
            'tech': ": {query}"
        }
    }
    return prompt_templates.get(lang, {}).get(domain, "{query}").format(query=query)