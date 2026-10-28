![banner](assets/brain4j-logo.svg)

# Brain4J

[![Documentation](https://img.shields.io/badge/documentation-wiki-ff5151?style=for-the-badge)](https://wiki.brain4j.org)
[![License](https://img.shields.io/github/license/brain4j-org/brain4j?style=for-the-badge&color=26ba43)](./LICENSE)
[![Discord](https://img.shields.io/discord/1367071190203498516?label=discord&style=for-the-badge&color=7289da)](https://discord.gg/SMKrxNjXBz)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/brain4j-org/brain4j?style=for-the-badge)](https://github.com/brain4j-org/brain4j/commits/3.0)

Brain4J is a lightweight, performant & open-source deep learning framework for Java.

Brain4J 3.0 requires Java 22 or newer.

Designed with portability and speed in mind, it's optimized for high performance, and it's ideal for those looking forward
to implement efficient and portable machine learning solutions in pure Java.

# Install

Brain4J is available on the [official repository](https://repo.brain4j.org/).

[![Latest Release](https://img.shields.io/badge/dynamic/xml?url=https://repo.brain4j.org/snapshots/org/brain4j/brain4j-core/maven-metadata.xml&query=/metadata/versioning/latest&label=Latest&color=blue&width=300&dpr=3&quality=100&sign=d974fb0d3aaf9afef9fcb212977ea237&sv=3)](https://repo.brain4j.org/#/snapshots/org/brain4j)

```groovy
repositories {
    mavenCentral()
    maven { url 'https://repo.brain4j.org/releases' }
    maven { url 'https://repo.brain4j.org/snapshots' }
}

dependencies {
    implementation 'org.brain4j:brain4j-core:$VERSION'
    implementation 'org.brain4j:brain4j-math:$VERSION'
}
```

See the [installation guide](https://github.com/brain4j-org/brain4j/wiki/Installation) for more information.

# Documentation

The 3.0 documentation can be found [here](https://docs.brain4j.org).

The GitHub wiki is **deprecated** and only contains the 2.0 documentation.

# Contributing

See the [contributing guide](./.github/CONTRIBUTING.md).

# Contact

This project is maintained by [xEcho1337](https://github.com/xEcho1337) and [Adversing](https://github.com/Adversing).

# License

Brain4J is licensed under the [Apache License, Version 2.0](./LICENSE).
