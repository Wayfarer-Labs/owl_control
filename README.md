# vg_control
Videogame control data collected via OBS + C input library

## Development

### Building and Releasing

This project uses GitHub Actions to automatically build and release the application when a new version is tagged.

To create a new release:

1. Run one of the following commands to bump the version:
   ```bash
   # For a patch version (1.0.0 -> 1.0.1)
   npm run version:patch

   # For a minor version (1.0.0 -> 1.1.0)
   npm run version:minor
   
   # For a major version (1.0.0 -> 2.0.0)
   npm run version:major
   ```

2. Push the changes and tags to GitHub:
   ```bash
   git push && git push --tags
   ```

3. GitHub Actions will automatically:
   - Build the application for Windows, macOS, and Linux
   - Create a new GitHub release with all distribution files
   - Generate release notes based on the commits since the last release

The built application will be available on the releases page of the repository.
