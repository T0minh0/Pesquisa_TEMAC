import spacy
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import requests

# Pipeline para coleta, pre-processamento e análise de texto completo

# 1. Coleta de dados via Google Scholar

# Usando SerpAPI, eles dão direito a 100 pesquisas por mês
def coletar_artigos(query, max_resultados=10, api_key="-------------------------------"):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": api_key,
        "num": max_resultados,
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        resultados = response.json().get("organic_results", [])
        artigos = []
        for resultado in resultados[:max_resultados]:
            artigo = {
                "titulo": resultado.get("title", "Título não disponível"),
                "resumo": resultado.get("snippet", "Resumo não disponível"),
                "link": resultado.get("link", ""),
            }
            artigos.append(artigo)
        return artigos
    else:
        print(f"Erro ao acessar SerpApi: {response.status_code}, {response.text}")
        return []

# 2. Pre-processamento de texto
# Configurar o modelo de NLP (SpaCy)
# Baixe os modelos com "python -m spacy download en_core_web_sm" e "python -m spacy download pt_core_news_sm"
nlp_en = spacy.load("en_core_web_sm")
nlp_pt = spacy.load("pt_core_news_sm")

# Gerar palavras de parada para português usando SpaCy
stop_words_pt = list(nlp_pt.Defaults.stop_words)

def preprocessar_texto(texto, idioma="en"):
    if idioma == "pt":
        doc = nlp_pt(texto)
    else:
        doc = nlp_en(texto)

    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# 3. Extração de palavras-chave
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

# 4. Relevância baseada em similaridade
def calcular_relevancia(artigos, termo_referencia, idioma="en"):
    stop_words = stop_words_pt if idioma == "pt" else 'english'
    textos = [artigo['resumo'] for artigo in artigos if artigo['resumo'] != "Resumo não disponível"]
    tfidf = TfidfVectorizer(stop_words=stop_words)

    # Adicionar o termo de referência como um documento adicional
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

# 6. Execução
if __name__ == "__main__":
    # Inicializar lista de artigos e a query acumulada
    artigos = []
    query_acumulada = ""

    # Solicitar idioma para análise
    idioma = input("Digite o idioma dos artigos (en para inglês, pt para português): ").strip().lower()
    if idioma not in ["en", "pt"]:
        print("Idioma inválido! Usando inglês como padrão.")
        idioma = "en"

    while True:
        # Solicitar entrada do usuário para o tema de busca ou extensão
        novo_termo = input("Digite um termo para pesquisa (ou 'sair' para finalizar): ").strip()
        if novo_termo.lower() == "sair":
            break

        # Atualizar a query acumulada
        if query_acumulada:
            query_acumulada += f" {novo_termo}"
        else:
            query_acumulada = novo_termo

        # Solicitar número máximo de artigos
        try:
            max_resultados = int(input("Digite o número máximo de artigos para coletar: "))
        except ValueError:
            print("Número inválido! Usando 10 artigos como padrão.")
            max_resultados = 10

        # Coletar artigos com a query acumulada
        novos_artigos = coletar_artigos(query=query_acumulada, max_resultados=max_resultados)

        # Verificar se há novos artigos
        if novos_artigos:
            print(f"Artigos encontrados para a pesquisa '{query_acumulada}':")

            # Ordenar novos artigos por relevância
            artigos_ordenados = calcular_relevancia(novos_artigos, query_acumulada, idioma=idioma)

            # Filtrar artigos relevantes
            artigos_relevantes = [artigo for artigo in artigos_ordenados if artigo['relevancia'] > 0]

            # Adicionar novos artigos à lista principal
            artigos.extend(artigos_relevantes)

            # Exibir os artigos relevantes encontrados
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
        else:
            print(f"Nenhum artigo encontrado para a pesquisa '{query_acumulada}'.")

    # Salvar todos os artigos coletados em JSON
    if artigos:
        salvar_artigos(artigos)
        print("Todos os artigos relevantes salvos com sucesso.")
    else:
        print("Nenhum artigo relevante foi coletado.")
