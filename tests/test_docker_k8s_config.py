from pathlib import Path
import yaml


def test_dockerfile_contract():
    content = Path("Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in content
    assert "EXPOSE 8000" in content
    assert "--host" in content and "0.0.0.0" in content
    assert "--port" in content and "8000" in content
    assert content.index("COPY requirements.txt") < content.index("RUN pip install")
    assert content.index("RUN pip install") < content.index("COPY . .")


def test_dockerignore_excludes_raw_data_and_secrets():
    lines = {
        line.strip()
        for line in Path(".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for required in ["data/raw", ".venv", "__pycache__", ".git", ".env"]:
        assert required in lines, f"Missing {required} in .dockerignore"
    assert "models" not in lines and "models/model.joblib" not in lines


def test_k8s_deployment_contract():
    doc = yaml.safe_load(Path("k8s/deployment.yaml").read_text(encoding="utf-8"))
    assert doc["kind"] == "Deployment"
    assert doc["metadata"]["name"] == "mp012-api"
    assert doc["spec"]["replicas"] == 3
    assert doc["spec"]["selector"]["matchLabels"] == doc["spec"]["template"]["metadata"]["labels"]

    container = doc["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "mp012-api:1.0"
    assert container["ports"][0]["containerPort"] == 8000

    readiness = container["readinessProbe"]
    liveness = container["livenessProbe"]
    assert readiness["httpGet"]["path"] == "/health"
    assert liveness["httpGet"]["path"] == "/health"
    assert liveness["initialDelaySeconds"] > readiness["initialDelaySeconds"]


def test_k8s_service_contract():
    deploy = yaml.safe_load(Path("k8s/deployment.yaml").read_text(encoding="utf-8"))
    svc = yaml.safe_load(Path("k8s/service.yaml").read_text(encoding="utf-8"))
    assert svc["kind"] == "Service"
    assert svc["metadata"]["name"] == "mp012-svc"
    assert svc["spec"]["selector"] == deploy["spec"]["template"]["metadata"]["labels"]
    port_cfg = svc["spec"]["ports"][0]
    assert port_cfg["port"] == 80
    assert port_cfg["targetPort"] == 8000
