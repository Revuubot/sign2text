import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get database URL from environment variable or use SQLite as fallback
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///asl_recognition.db")

# Handle SQLAlchemy 1.4+ compatibility with PostgreSQL URLs
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine
engine = create_engine(DATABASE_URL)

# Create base class for models
Base = declarative_base()

# Define models
class Recognition(Base):
    """
    Database model for storing ASL sign recognition events.
    """
    __tablename__ = 'recognitions'
    
    id = Column(Integer, primary_key=True)
    letter = Column(String(10), nullable=False)  # The recognized ASL letter
    confidence = Column(Float, nullable=False)   # Confidence score of the prediction
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_feedback = Column(String(50), nullable=True)  # User feedback on recognition (correct/incorrect)
    image_data = Column(Text, nullable=True)  # Base64 encoded image data (optional)
    
    def __repr__(self):
        return f"<Recognition(id={self.id}, letter='{self.letter}', confidence={self.confidence:.2f})>"

# Create session factory
Session = sessionmaker(bind=engine)

def initialize_database():
    """
    Initialize the database by creating all tables.
    """
    try:
        Base.metadata.create_all(engine)
        print(f"Database initialized successfully using {DATABASE_URL}")
        return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

# Initialize the database when this module is imported
initialize_database()

def save_recognition(letter, confidence, image_data=None):
    """
    Save a recognition event to the database.
    
    Args:
        letter: The recognized ASL letter
        confidence: Confidence score of the prediction
        image_data: Optional base64 encoded image data
    
    Returns:
        The created Recognition object
    """
    session = Session()
    try:
        # Create new recognition record
        recognition = Recognition(
            letter=letter,
            confidence=confidence,
            image_data=image_data
        )
        
        # Add to session and commit
        session.add(recognition)
        session.commit()
        
        return recognition
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_recent_recognitions(limit=20):
    """
    Get the most recent recognition events.
    
    Args:
        limit: Maximum number of records to return
        
    Returns:
        List of Recognition objects
    """
    session = Session()
    try:
        # Query the most recent recognitions
        recognitions = session.query(Recognition).order_by(
            Recognition.timestamp.desc()
        ).limit(limit).all()
        
        return recognitions
    except Exception as e:
        print(f"Error retrieving recognitions: {e}")
        return []
    finally:
        session.close()

def get_recognition_stats():
    """
    Get statistics about recognition events.
    
    Returns:
        Dictionary with statistics
    """
    session = Session()
    try:
        # Get total count
        total_count = session.query(func.count(Recognition.id)).scalar()
        
        if total_count == 0:
            return {'total_count': 0, 'by_letter': []}
        
        # Get stats by letter
        letter_stats = session.query(
            Recognition.letter,
            func.count(Recognition.id).label('count'),
            func.avg(Recognition.confidence).label('avg_confidence')
        ).group_by(Recognition.letter).all()
        
        # Convert to dictionary format
        stats_by_letter = [
            {
                'letter': stat.letter,
                'count': stat.count,
                'avg_confidence': float(stat.avg_confidence)
            }
            for stat in letter_stats
        ]
        
        # Get feedback stats
        feedback_counts = session.query(
            Recognition.user_feedback,
            func.count(Recognition.id).label('count')
        ).filter(Recognition.user_feedback != None).group_by(
            Recognition.user_feedback
        ).all()
        
        feedback_stats = [
            {
                'feedback': stat.user_feedback or "None",
                'count': stat.count
            }
            for stat in feedback_counts
        ]
        
        # Return combined stats
        return {
            'total_count': total_count,
            'by_letter': stats_by_letter,
            'by_feedback': feedback_stats
        }
    except Exception as e:
        print(f"Error retrieving statistics: {e}")
        return {'total_count': 0, 'by_letter': [], 'by_feedback': []}
    finally:
        session.close()

def save_user_feedback(recognition_id, feedback):
    """
    Save user feedback on a recognition event.
    
    Args:
        recognition_id: ID of the recognition
        feedback: User feedback (e.g., 'correct', 'incorrect')
        
    Returns:
        True if successful, False otherwise
    """
    session = Session()
    try:
        # Find the recognition by ID
        recognition = session.query(Recognition).filter_by(id=recognition_id).first()
        
        if not recognition:
            print(f"Recognition with ID {recognition_id} not found")
            return False
        
        # Update the feedback
        recognition.user_feedback = feedback
        session.commit()
        
        return True
    except Exception as e:
        session.rollback()
        print(f"Error saving feedback: {e}")
        return False
    finally:
        session.close()