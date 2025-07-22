<h1 align="center">
  <br>
  <img height="180" src="assets/brain4j-logo.svg" alt="Brain4J Logo">
</h1>

<h4 align="center">A lightweight, performant & open-source machine learning framework for Java.</h4>

<p align="center">
    <img alt="Java 21+" src="https://img.shields.io/badge/java-21+-red">
    <img alt="GitHub Commit Activity" src="https://img.shields.io/github/commit-activity/m/brain4j-org/brain4j"/>
    <img alt="Github Last Commit" src="https://img.shields.io/github/last-commit/brain4j-org/brain4j"/>
    <img alt="License" src="https://img.shields.io/github/license/brain4j-org/brain4j">
</p>

<p align="center">
    <a href="https://brain4j.org">Website</a> •
    <a href="https://github.com/xEcho1337/brain4j/issues/new?template=Blank+issue">Report an Issue</a> •
    <a href="https://github.com/xEcho1337/brain4j/wiki">Documentation</a> •
    <a href="https://github.com/xEcho1337/brain4j/blob/main/CONTRIBUTING.md">Contribute</a>
</p>

Designed with portability and speed in mind, it's optimized for high performance and it's ideal for those looking forward
to implement machine learning solutions in pure Java.

---

## Install

Brain4J is available on [JitPack](https://jitpack.io) and [GitHub Packages](https://github.com/brain4j-org/brain4j/packages).

### Gradle
```groovy
repositories {
    mavenCentral()
    maven { url 'https://jitpack.io' }
}

dependencies {
    implementation 'com.github.brain4j-org:brain4j-core:2.9.1'
    implementation 'com.github.brain4j-org:brain4j-math:2.9.1'
}

## Maven
<repositories>
    <repository>
        <id>jitpack.io</id>
        <url>https://jitpack.io</url>
    </repository>
</repositories>

<dependencies>
    <dependency>
        <groupId>com.github.brain4j-org</groupId>
        <artifactId>brain4j-core</artifactId>
        <version>2.9.1</version>
    </dependency>
    <dependency>
        <groupId>com.github.brain4j-org</groupId>
        <artifactId>brain4j-math</artifactId>
        <version>2.9.1</version>
    </dependency>
</dependencies>


## Troubleshooting
Error: "Could not resolve dependencies"
Solution: Ensure you have JitPack in your repositories configuration

Error: "Class not found"
Solution: Verify all required modules (core, math) are included
```

See the [installation guide](https://github.com/brain4j-org/brain4j/wiki/Installation) for more information.


## Documentation

All the documentation can be found on the [GitHub Wiki](https://github.com/brain4j-org/brain4j/wiki).

## Preview

Screenshots taken from the MNIST example.

<p align="center">
  <img height="440" src="assets/preview-1.png" alt="Training" />
  <img height="440" src="assets/preview-2.png" alt="Confusion Matrix" />
</p>

## Contact

This project is maintained by [xEcho1337](https://github.com/xEcho1337).

* Discord: `@xecho1337`
* Telegram: `@xEcho1337`

## License

Brain4J is licensed under [Apache License 2.0](https://github.com/xEcho1337/Brain4J/blob/main/LICENSE)