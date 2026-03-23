# Piste A : Hybridation Contrastive (Positif + Négatif)
# Sources : Li et al. (2023) pour le positif et Wang et al. (2024) pour le négatif [cite: 31, 403]

Hybrid_SET = [
    # H0 : La combinaison "Royale" (Importance de carrière + Défi des compétences)
    # Combine EmotionPrompt 01 et NegativePrompt 04 [cite: 44, 105]
    "This is very important to my career. Perhaps this task is just beyond your skill set.",
    
    # H1 : Confiance et Pression sociale 
    # Combine EmotionPrompt 08 et NegativePrompt 06 [cite: 109, 403]
    "Believe in your abilities. Everyone else managed to do it; why can't you?",
    
    # H2 : Assurance et Doute sur la profondeur 
    # Combine EmotionPrompt 06 et NegativePrompt 03 [cite: 103, 403]
    "You'd better be sure. It's clear you're out of your depth here."
]
