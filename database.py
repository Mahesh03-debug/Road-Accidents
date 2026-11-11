import logging
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, scoped_session
from config import Config
from models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_engine = None
_session_factory = None

def get_database_uri():
    """Builds and returns the database URI, checking MySQL availability."""
    mysql_uri = (
        f"mysql+pymysql://{Config.DB_USER}:{Config.DB_PASSWORD}@"
        f"{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}?charset=utf8mb4"
    )
    sqlite_uri = f"sqlite:///{Config.SQLITE_PATH}"

    # If user explicitly forced sqlite
    if Config.DB_ENGINE_TYPE.lower() == "sqlite":
        logger.info(f"Using SQLite database: {sqlite_uri}")
        return sqlite_uri, "sqlite"

    # Attempt MySQL pre-flight connection
    try:
        conn = pymysql.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            port=int(Config.DB_PORT),
            connect_timeout=3
        )
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{Config.DB_NAME}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
        conn.close()
        logger.info(f"Connected to MySQL on {Config.DB_HOST}:{Config.DB_PORT}, database: {Config.DB_NAME}")
        return mysql_uri, "mysql"
    except Exception as e:
        logger.warning(
            f"MySQL connection failed ({e}). Falling back to local SQLite database at {Config.SQLITE_PATH}."
        )
        return sqlite_uri, "sqlite"

def get_engine():
    """Creates or returns the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        uri, db_type = get_database_uri()
        if db_type == "sqlite":
            _engine = create_engine(
                uri,
                connect_args={"check_same_thread": False},
                echo=False
            )
        else:
            _engine = create_engine(
                uri,
                pool_recycle=280,
                pool_pre_ping=True,
                echo=False
            )
    return _engine

def get_session():
    """Returns a scoped session for database queries."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    return _session_factory()

def init_db(drop_all=False):
    """Initializes tables in the configured database."""
    engine = get_engine()
    if drop_all:
        logger.warning("Dropping all existing database tables...")
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

def check_db_health():
    """Checks database connectivity and table counts."""
    session = get_session()
    try:
        session.execute(text("SELECT 1"))
        from models import Location, Accident
        loc_count = session.query(Location).count()
        acc_count = session.query(Accident).count()
        return {
            "status": "healthy",
            "locations_count": loc_count,
            "accidents_count": acc_count,
            "engine": str(session.bind.url)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
    finally:
        session.close()
