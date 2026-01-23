#!/usr/bin/env python3
"""
Create a demo user for public ngrok sharing
This user has limited permissions for security
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

os.environ["DATABASE_URL"] = "sqlite:///./2ndbrain_ucla.db"

from backend.database.database import SessionLocal
from backend.database.models import User, Tenant, UserRole
import bcrypt


def create_demo_user():
    """Create a demo user with viewer permissions"""

    db = SessionLocal()

    try:
        # Get the UCLA tenant
        tenant = db.query(Tenant).filter_by(slug="ucla-beat-healthcare").first()

        if not tenant:
            print("❌ UCLA BEAT Healthcare tenant not found!")
            print("   Make sure you've imported the data first.")
            return False

        # Check if demo user already exists
        demo_user = db.query(User).filter_by(
            tenant_id=tenant.id,
            email="demo@ucla.beat"
        ).first()

        if demo_user:
            print("ℹ️  Demo user already exists")
            print(f"   Email: demo@ucla.beat")
            print(f"   Role: {demo_user.role.value}")
            return True

        # Hash password
        password = "DemoUCLA2024"
        salt = bcrypt.gensalt(rounds=12)
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

        # Create demo user with viewer role (read-only)
        demo_user = User(
            tenant_id=tenant.id,
            email="demo@ucla.beat",
            name="Demo Viewer",
            role=UserRole.VIEWER,
            is_active=True,
            password_hash=password_hash
        )

        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)

        print("✅ Demo user created successfully!")
        print("")
        print("📋 Demo User Credentials:")
        print("   Email: demo@ucla.beat")
        print("   Password: DemoUCLA2024")
        print("   Role: VIEWER (read-only)")
        print("")
        print("🔒 Security:")
        print("   ✅ Cannot delete documents")
        print("   ✅ Cannot modify settings")
        print("   ✅ Can only view and search")
        print("")
        print("💡 Share these credentials instead of admin credentials!")

        return True

    except Exception as e:
        print(f"❌ Error creating demo user: {e}")
        db.rollback()
        import traceback
        traceback.print_exc()
        return False

    finally:
        db.close()


if __name__ == "__main__":
    print("🎓 UCLA 2nd Brain - Demo User Setup")
    print("====================================")
    print("")

    success = create_demo_user()
    sys.exit(0 if success else 1)
