from neo4j import GraphDatabase
from src.core.config import settings

class Neo4jClient:
    def __init__(self, uri=None, username=None, password=None):
        self.uri = uri or settings.NEO4J_URI
        self.username = username or settings.NEO4J_USERNAME
        self.password = password or settings.NEO4J_PASSWORD
        self.driver = None
        self.connect()

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            self.driver.verify_connectivity()
            print("[Neo4j] Connected to database successfully.", flush=True)
        except Exception as e:
            print(f"[Neo4j] Connection failed: {e}", flush=True)
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, query: str, parameters: dict = None) -> list[dict]:
        if not self.driver:
            print("[Neo4j] Driver not initialized, query aborted.", flush=True)
            return []
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"[Neo4j] Query execution failed: {e}", flush=True)
            return []