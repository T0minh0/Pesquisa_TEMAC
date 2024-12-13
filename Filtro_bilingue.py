import spacy
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scholarly import scholarly
import json
import os
import requests
# Pipeline para coleta, pre-processamento e analise de texto completo

# 1. Coleta de dados via Google Scholar

# Funcao para buscar artigos usando a biblioteca scholarly
def coletar_artigos(query, max_resultados=10):
    resultados = scholarly.search_pubs(query)
    artigos = []

    for i, resultado in enumerate(resultados):
        if i >= max_resultados:
            break
        artigo = {
            "titulo": resultado.get("bib", {}).get("title", "Título não disponível"),
            "resumo": resultado.get("bib", {}).get("abstract", "Resumo não disponível"),
            "link": resultado.get("eprint_url", "")
        }
        artigos.append(artigo)

    return artigos

# 2. Pre-processamento de texto
# Configurar o modelo de NLP (SpaCy)
# Baixe os modelos com "python -m spacy download en_core_web_sm" e "python -m spacy download pt_core_news_sm"
nlp_en = spacy.load("en_core_web_sm")
nlp_pt = spacy.load("pt_core_news_sm")

# Gerar palavras de parada para portugues usando SpaCy
stop_words_pt = list(nlp_pt.Defaults.stop_words)

def preprocessar_texto(texto, idioma="en"):

    if idioma == "pt":
        doc = nlp_pt(texto)
    else:
        doc = nlp_en(texto)

    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# 3. Extracao de palavras-chave
# Usando YAKE
def extrair_palavras_chave_yake(texto, max_palavras=10, idioma="en"):

    kw_extractor = yake.KeywordExtractor(lan=idioma, top=max_palavras)
    keywords = kw_extractor.extract_keywords(texto)
    return keywords

# Usando TF-IDF
def extrair_palavras_chave_tfidf(textos, max_features=10):

    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    matriz_tfidf = tfidf.fit_transform(textos)
    palavras_chave = tfidf.get_feature_names_out()
    return palavras_chave

# 4. Relevancia baseada em similaridade
def calcular_relevancia(artigos, termo_referencia, idioma="en"):

    stop_words = stop_words_pt if idioma == "pt" else 'english'
    textos = [artigo['resumo'] for artigo in artigos if artigo['resumo'] != "Resumo não disponível"]
    tfidf = TfidfVectorizer(stop_words=stop_words)

    # Adicionar o termo de referencia como um documento adicional
    matriz_tfidf = tfidf.fit_transform(textos + [termo_referencia])
    
    # Calcular similaridade do termo com cada texto
    similaridades = cosine_similarity(matriz_tfidf[-1], matriz_tfidf[:-1]).flatten()
    
    # Adicionar as similaridades aos artigos
    for i, artigo in enumerate(artigos):
        artigo['relevancia'] = similaridades[i] if i < len(similaridades) else 0

    # Ordenar os artigos pela relevância
    artigos_ordenados = sorted(artigos, key=lambda x: x['relevancia'], reverse=True)
    return artigos_ordenados

# 5. Salvar artigos localmente
def salvar_artigos(artigos, caminho="artigos_relevantes.json"):

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(artigos, f, ensure_ascii=False, indent=4)

def baixar_pdf(artigo, pasta_destino="pdfs"):

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    link = artigo.get("link")
    if link:
        try:
            response = requests.get(link, stream=True)
            if response.status_code == 200:
                arquivo_pdf = os.path.join(pasta_destino, f"{artigo['titulo'][:50]}.pdf")
                with open(arquivo_pdf, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"Erro ao baixar PDF: {e}")
    return False

# 6. Execucao
if __name__ == "__main__":
    # Solicitar entrada do usuario para o tema de busca
    tema = input("Digite o tema que deseja pesquisar: ")

    # Solicitar idioma para analise
    idioma = input("Digite o idioma dos artigos (en para inglês, pt para português): ").strip().lower()
    if idioma not in ["en", "pt"]:
        print("Idioma inválido! Usando inglês como padrão.")
        idioma = "en"

    # Solicitar entrada do usuario para o numero de artigos
    try:
        max_resultados = int(input("Digite o numero maximo de artigos para coletar: "))
    except ValueError:
        print("Número inválido! Usando 10 artigos como padrão.")
        max_resultados = 10

    # Termo de referencia para relevancia (pode ser o mesmo que o tema)
    termo_referencia = tema

    # Coletar artigos com o tema fornecido
    artigos = coletar_artigos(query=tema, max_resultados=max_resultados)

    # Verificar artigos encontrados
    if artigos:
        print("Artigos encontrados:")

        # Ordenar artigos por relevancia
        artigos_ordenados = calcular_relevancia(artigos, termo_referencia, idioma=idioma)

        # Filtrar artigos com relevancia
        artigos_relevantes = [artigo for artigo in artigos_ordenados if artigo['relevancia'] > 0]

        # Exibir artigos relevantes
        for artigo in artigos_relevantes:
            print(f"Título: {artigo['titulo']}")
            print(f"Relevância: {artigo['relevancia']:.2f}")
            print(f"Link: {artigo['link']}")

            # Extrair e imprimir palavras-chave
            palavras_chave = extrair_palavras_chave_yake(artigo['resumo'], idioma=idioma)
            print("Palavras-chave extraídas:")
            for palavra, score in palavras_chave:
                print(f"  {palavra} (score: {score:.4f})")

            # Tentar baixar PDF
            if baixar_pdf(artigo):
                print("PDF baixado com sucesso.")
            else:
                print("Não foi possível baixar o PDF.")

        # Salvar artigos relevantes em JSON
        salvar_artigos(artigos_relevantes)
        print("Artigos relevantes salvos com sucesso.")
    else:
        print("Nenhum artigo encontrado com o tema fornecido.")