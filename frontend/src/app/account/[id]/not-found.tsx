'use client'
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";

export default function ProfileNotFound() {

  return (
    <Container sx={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      width: '100%',
    }}>
      <Typography>Profile not found</Typography>
    </Container>
  );
}