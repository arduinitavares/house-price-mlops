# Etapa base
FROM python:3.10-slim

# Diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos do projeto para o container
COPY . .

# Instala dependências
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expõe a porta onde o uvicorn irá rodar
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]