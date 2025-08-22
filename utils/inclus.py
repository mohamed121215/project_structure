def inclusion_score1(a, b):
    """
    Calcule le score d'inclusion de la chaîne a dans la chaîne b.
    Score = pourcentage de caractères de a trouvés dans b dans le même ordre.
    """
    a = a.lower()
    b = b.lower()
    if a in b:
        return True
    else:
        return False