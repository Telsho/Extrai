from extrai.core.code_generation.python_builder import ImportManager

class TestImportManager:
    def test_malformed_imports_fallback(self):
        """Tests that malformed imports are handled gracefully by falling back to appending as-is."""
        manager = ImportManager()
        
        # Add malformed imports that should trigger fallback behavior
        malformed_sqlmodel = "from sqlmodel" # Missing " import "
        malformed_typing = "from typing"     # Missing " import "
        malformed_import = "import ,"        # Missing modules (strips to "import ,")
        
        manager.add_custom_imports([malformed_sqlmodel, malformed_typing, malformed_import])
        
        rendered = manager.render()
        
        assert "from sqlmodel" in rendered
        assert "from typing" in rendered
        
        # Check that "import ," line is present exactly as added
        rendered_lines = rendered.split('\n')
        assert "import ," in rendered_lines
