'use client'

import Typography from "@mui/material/Typography";
import {Grid} from "@mui/material";
import Container from "@mui/material/Container";
import Box from "@mui/material/Box";
import {api} from "~/trpc/react";
import {useSession} from "next-auth/react";
import ProfileCard from "~/app/_components/profile-card";
import {notFound} from "next/navigation";
import AccountLoading from "~/app/profile/[id]/loading";

interface AccountPageProps {
    params: { id: string };
}

export default function Account({ params }: AccountPageProps) {
    const { data: session, status } = useSession();
    const { data: account, isLoading} = api.user.byId.useQuery(params.id);

    if (!account){
        if (!isLoading){
            return notFound()
        } else {
            return <AccountLoading/>
        }
    }

    const boxStyle = {
        borderRadius: 2,
        overflow: 'hidden',
        boxShadow: 3,
        bgcolor: 'background.paper',
        width: {
            xs: '100%',
            md: '95%',
        },
        p: 4,
        margin: 'auto',
        flex: 1,
    };

    return (
      <Container maxWidth={"xl"} sx={{
          minHeight: '100vh',
      }}>
          <Grid container columns={14} sx={{
              paddingTop: {
                  xs: 2,
                  md: 3,
              },
              height: '100vh',
          }}
                spacing={{
                    xs: 3,
                    md: 2,
                }}
          >
              <Grid item xs={16} md={4} sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '1.25rem',
              }}>
                  <ProfileCard account={account} />

                  <Box sx={boxStyle}>
                      <Typography component={'h2'} variant={'h5'}>
                          Friend&apos;s Activity
                      </Typography>
                  </Box>
              </Grid>
              <Grid item xs={16} md={6} sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '1.25rem',
              }}>
                  <Box sx={boxStyle}>
                      <Typography component={'h1'} variant={'h5'}>
                          Your Profile
                      </Typography>

                      <Typography component={'h2'} variant={'h6'} sx={{
                          fontWeight: 700,
                      }}>
                          Favorites
                      </Typography>

                      <Typography component={'h2'} variant={'h6'} sx={{
                          fontWeight: 700,
                      }}>
                          Recents
                      </Typography>
                  </Box>
              </Grid>

              <Grid item xs={16} md={4} sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '1.25rem',
              }}>
                  <Box sx={boxStyle}>
                      <Typography component={'h2'} variant={'h5'}>
                          Friend Requests
                      </Typography>
                  </Box>
              </Grid>
          </Grid>
      </Container>
    );
}