#!/usr/bin/env python3
"""
Import JSONL documents into 2nd Brain system
Usage: python import_jsonl.py <jsonl_file_path>
"""

import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.database.database import SessionLocal, init_db
from backend.database.models import Tenant, User, Document, DocumentStatus, ConnectorType, UserRole


def hash_content(content: str) -> str:
    """Generate SHA-256 hash of content for deduplication"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_or_create_tenant(db, tenant_name: str = "Default Organization"):
    """Get or create a tenant"""
    # Create slug from name
    slug = tenant_name.lower().replace(" ", "-")

    tenant = db.query(Tenant).filter(Tenant.slug == slug).first()
    if not tenant:
        tenant = Tenant(
            name=tenant_name,
            slug=slug,
            plan="professional",
            document_limit=10000
        )
        db.add(tenant)
        db.commit()
        db.refresh(tenant)
        print(f"✅ Created tenant: {tenant.name} ({tenant.id})")
    else:
        print(f"✅ Using existing tenant: {tenant.name} ({tenant.id})")

    return tenant


def get_or_create_user(db, tenant_id, email: str = "admin@2ndbrain.local"):
    """Get or create a user"""
    user = db.query(User).filter(
        User.tenant_id == tenant_id,
        User.email == email
    ).first()

    if not user:
        user = User(
            tenant_id=tenant_id,
            email=email,
            name="System Admin",
            role=UserRole.OWNER,
            is_active=True
        )
        user.set_password("admin123")  # Default password
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✅ Created user: {user.email} ({user.id})")
        print(f"   Default password: admin123")
    else:
        print(f"✅ Using existing user: {user.email} ({user.id})")

    return user


def import_jsonl(jsonl_path: str, tenant_name: str = "UCLA BEAT Healthcare"):
    """Import documents from JSONL file"""

    # Initialize database
    print("🔧 Initializing database...")
    init_db()

    db = SessionLocal()

    try:
        # Get or create tenant and user
        tenant = get_or_create_tenant(db, tenant_name)
        user = get_or_create_user(db, tenant.id)

        # Read JSONL file
        print(f"\n📄 Reading JSONL file: {jsonl_path}")
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()

        print(f"   Found {len(lines)} document(s)")

        # Import each document
        imported = 0
        skipped = 0

        for idx, line in enumerate(lines, 1):
            doc_data = json.loads(line.strip())

            # Extract fields
            file_name = doc_data.get('file_name', 'Untitled')
            content = doc_data.get('content', '')
            metadata = doc_data.get('metadata', {})
            project = doc_data.get('project', 'General')

            # Generate content hash for deduplication
            content_hash = hash_content(content)

            # Check if document already exists
            existing = db.query(Document).filter(
                Document.tenant_id == tenant.id,
                Document.content_hash == content_hash
            ).first()

            if existing:
                print(f"⏭️  [{idx}/{len(lines)}] Skipped (duplicate): {file_name}")
                skipped += 1
                continue

            # Create new document
            document = Document(
                tenant_id=tenant.id,
                created_by=user.id,
                title=file_name,
                content=content,
                content_hash=content_hash,
                source_type=ConnectorType.UPLOAD,
                source_id=f"jsonl_import_{idx}",
                doc_metadata={
                    'original_metadata': metadata,
                    'project': project,
                    'import_date': datetime.utcnow().isoformat(),
                    'import_source': 'jsonl_import'
                },
                status=DocumentStatus.PENDING,
                classification='work',
                classification_confidence=100,
                cluster_label=project
            )

            db.add(document)
            imported += 1
            print(f"✅ [{idx}/{len(lines)}] Imported: {file_name}")

        # Commit all documents
        db.commit()

        print(f"\n✨ Import complete!")
        print(f"   Imported: {imported} document(s)")
        print(f"   Skipped:  {skipped} duplicate(s)")
        print(f"\n📊 Next steps:")
        print(f"   1. Run the backend: cd backend && python -m api.app")
        print(f"   2. Login with: {user.email} / admin123")
        print(f"   3. Navigate to Documents page to view imported content")
        print(f"   4. Run 'Rebuild Index' to enable RAG search")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_jsonl.py <jsonl_file_path> [tenant_name]")
        print("\nExample:")
        print("  python import_jsonl.py ~/Downloads/data.jsonl 'UCLA BEAT Healthcare'")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    tenant_name = sys.argv[2] if len(sys.argv) > 2 else "UCLA BEAT Healthcare"

    if not Path(jsonl_path).exists():
        print(f"❌ File not found: {jsonl_path}")
        sys.exit(1)

    import_jsonl(jsonl_path, tenant_name)
